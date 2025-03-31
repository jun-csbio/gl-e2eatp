#  ************************************************************************
#  * The following codes is only free for academic use.                   *
#  *                                                                      *
#  * If you need it for a commercial use, email Jun Hu (junh_cs@126.com). *
#  ************************************************************************

import argparse
import os

import numpy as np
import torch
from torch import nn

import esm


def createFolder(folder):
    if not exists(folder):
        os.makedirs(folder)


def print_namespace(anamespace, ignore_none=True):
    for key in anamespace.__dict__:
        if ignore_none and None is anamespace.__dict__[key]:
            continue
        print("{}: {}".format(key, anamespace.__dict__[key]))


def exists(fileOrFolderPath):
    return os.path.exists(fileOrFolderPath)


def loadFasta(fasta):
    with open(fasta, 'r') as f:
        lines = f.readlines()
    ans = {}
    name = ''
    seq_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if 1 < len(name):
                ans[name] = "".join(seq_list)
            name = line[1:]
            seq_list = []
        else:
            seq_list.append(line)
    if 0 < seq_list.__len__():
        ans[name] = "".join(seq_list)
    return ans


def load_npz(filename):
    data = dict(np.load(filename, allow_pickle=True))
    return data['x']


def pick_up_one_sliding_window(emb, mid_ind, winsize=17):
    """
    :param emb: (tensor)[L, D]
    :param mid_ind: the position index
    :return (tensor)[winsize, D]
    """
    seq_len, emb_dim = emb.size()
    half_winsize = winsize // 2

    ans = torch.zeros(winsize, emb_dim).to(emb.device)

    ans_ind = 0
    for ind in range(mid_ind - half_winsize, mid_ind + half_winsize + 1):
        if 0 <= ind < seq_len:
            ans[ans_ind] = emb[ind]
        ans_ind += 1

    return ans


class JSeq2ESM2:
    def __init__(self, esm2_model_path, device='cpu'):
        with torch.no_grad():
            self.esm2, alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_model_path)
            self.tokenizer = alphabet.get_batch_converter()
            del alphabet

            self.esm2 = self.esm2.to(device)
            for param in self.esm2.parameters():
                param.requires_grad = False
            self.esm2.eval()

        self.device = device
        self.emb_dim = self.esm2.embed_dim
        self.layer_num = self.esm2.num_layers

    def tokenize(self, seq):
        """
        :param tuple_list: e.g., [('seq1', 'FFFFF'), ('seq2', 'AAASDA')]
        """
        tuple_list = [("seq", "{}".format(seq))]
        with torch.no_grad():
            _, _, tokens = self.tokenizer(tuple_list)
            return tokens.to(self.device)

    def embed(self, seq):
        with torch.no_grad():
            if len(seq) < 5000:
                # [B, L_rec, D]
                return self.esm2(self.tokenize(seq),
                                 repr_layers=[self.layer_num])["representations"][self.layer_num][..., 1:-1, :]
            else:
                embs = None
                for ind in range(0, len(seq), 5000):
                    sind = ind
                    eind = min(ind+5000, len(seq))
                    sub_seq = seq[sind:eind]
                    print(len(sub_seq), len(seq))
                    sub_emb = self.esm2(self.tokenize(sub_seq),
                                        repr_layers=[self.layer_num])["representations"][self.layer_num][..., 1:-1, :]
                    if None is embs:
                        embs = sub_emb
                    else:
                        embs = torch.cat([embs, sub_emb], dim=1)
                print(embs.size())
                return embs


class AttnPooling(nn.Module):
    """
    e.g.,
        B = 5
        L = 196
        D = 128
        l = 1
        head_num = 8
        x = torch.randn(B, L, D)
        obj = AttnPooling(D, l, head_num)
        O = obj(x)
        print(O.shape)
        exit()
    """
    def __init__(self, dim, out_len, head_num):
        super(AttnPooling, self).__init__()
        self.out_len = out_len
        self.Q = nn.Parameter(torch.randn(out_len, dim))
        self.mattn = nn.MultiheadAttention(embed_dim=dim, num_heads=head_num)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        :param x: [B, L, dim(D)]
        :return: [B, out_len(l), dim(D)]
        """
        B, _, _ = x.shape

        # [l, D] --> [l, B, D]
        Q = self.Q.unsqueeze(1).expand(-1, B, -1)

        # [B, L, D] --> [L, B, D]
        x = x.permute(1, 0, 2)

        # -->[l, B, D]
        out, _ = self.mattn(Q, x, x)

        # [l, B, D] --> [B, l, D]
        out = out.permute(1, 0, 2)
        out = self.ln(out)
        return out


class JActivator(nn.Module):
    def __init__(self, name='relu'):
        super(JActivator, self).__init__()
        self.obj = None
        if name == 'relu':
            self.obj = nn.ReLU()
        elif name == 'gelu':
            self.obj = nn.GELU()
        elif name == 'silu':
            self.obj = nn.SiLU()
        elif name == 'prelu':
            self.obj = nn.PReLU()
        elif name == 'sig':
            self.obj = nn.Sigmoid()
        elif name == 'tanh':
            self.obj = nn.Tanh()

    def forward(self, x):
        if None is self.obj:
            return x
        return self.obj(x)


class JCrossOutProduct(nn.Module):
    def __init__(
            self,
            in_dim,
            mid_dim,
            out_dim
    ):
        super().__init__()
        self.l = nn.Linear(in_dim, mid_dim)
        self.lo = nn.Linear(mid_dim, out_dim)

    def _opm(self, a, b):
        """
        :param a: [B, L, D_mid]
        :param b: [B, l, D_mid]
        :return: [B, L, l, D_out]
        """

        # [B, L, l, D_mid]
        outer = torch.einsum("...bc,...dc->...bdc", a, b)

        # [B, L, l, D_out]
        outer = self.lo(outer)

        return outer

    def forward(
            self,
            x,
    ):
        """
        :param x: [B, L, D]
        :return: [B, L, L, d]
        """
        a = self.l(x)
        o = self._opm(a, a)
        return o


class JResCOPAttn(nn.Module):
    def __init__(self, dim, activator_name='none', dropout=0.3):
        super().__init__()
        self.cop = JCrossOutProduct(dim, dim, dim)
        self.l = nn.Linear(dim, dim)

        self.shadow = nn.Sequential(
            nn.Dropout(p=dropout),
            JActivator(name=activator_name),
            nn.LayerNorm(dim),
        )

    def forward(self, x, mask=None):
        """
        :param x: [B, L, D]
        :param mask: [B, L, L]
        :return:
        """
        # [B, L, D] --> [B, L, L, d]
        tm = self.cop(x)
        if None is not mask:
            mask = mask.unsqueeze(-1)
            tm = tm.masked_fill(mask == 0, 0.)

        # [B, L, D] --> [B, L, d]
        tx = self.l(x)

        # [B, L, L, d], [B, L, d] --> [B, L, d]
        x = x + torch.einsum("...cab,...ab->...cb", tm, tx)
        x = self.shadow(x)
        return x


class JResSelfAttn(nn.Module):
    def __init__(self, num_heads, dim, activator_name='none', dropout=0.3):
        super(JResSelfAttn, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.attn_shadow = nn.Sequential(
            nn.Dropout(p=dropout),
            JActivator(name=activator_name),
            nn.LayerNorm(dim),
        )

    def forward(self, x, mask=None):
        """
        :param x: [B, L, D]
        :param mask: [B*H, L, l] dtype = bool or float
        """
        if None is mask:
            tx, _ = self.attn(x, x, x)
        else:
            tx, _ = self.attn(x, x, x, attn_mask=mask)

        x = self.attn_shadow(x + tx)
        return x


class JCombineAttn(nn.Module):
    def __init__(self, num_heads, dim, core_num=6, activator_name='none', dropout=0.3):
        super().__init__()

        self.copattns = nn.ModuleList([
            JResCOPAttn(dim, activator_name, dropout) for _ in range(core_num)
        ])

        self.mattns = nn.ModuleList([
            JResSelfAttn(num_heads, dim, activator_name, dropout) for _ in range(core_num)
        ])

        self.l1 = nn.Linear(dim, dim)
        self.core_num = core_num

    def forward(self, x):
        for i in range(self.core_num):
            x1 = self.copattns[i](x)
            x2 = self.mattns[i](x)
            x = x + x1 + x2
        return self.l1(x)


class JResNet(nn.Module):
    def __init__(
            self,
            dim,
            core_num
    ):
        super().__init__()
        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(dim)
            ) for _ in range(core_num)
        ])

    def forward(
            self,
            x
    ):
        """
        :param x: [B, C, L]
        :return:
        """
        for b in self.body:
            x = x + b(x)
        return x


class JGlobalSeqFea(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, out_len, head_num=8, core_num=16):
        super().__init__()

        self.cnn1 = nn.Conv1d(in_dim, mid_dim, kernel_size=3, padding=1)
        self.resnet = JResNet(mid_dim, core_num)
        self.cnn2 = nn.Conv1d(mid_dim, out_dim, kernel_size=3, padding=1)

        self.attn_pool = AttnPooling(out_dim, out_len, head_num)
        self.l1 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """
        :param x: [B, L, C]
        """

        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.resnet(x)
        x = self.cnn2(x)
        x = x.permute(0, 2, 1)

        x = self.attn_pool(x)
        x = self.l1(x)

        return x


class JLocalSeqFea(nn.Module):
    def __init__(self,  in_dim, mid_dim, out_dim, head_num=8, core_num=16, activator_name='gelu'):
        super().__init__()

        self.l1 = nn.Linear(in_dim, mid_dim)
        self.comattn = JCombineAttn(head_num, mid_dim, core_num, activator_name)
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        """
        :param x: [B, l, d]
        """
        x = self.l1(x)
        x = self.comattn(x)
        x = self.l2(x)

        return x


class ModelCore(nn.Module):
    """
    in_dim = 128
    mid_dim = 64
    out_dim = 2
    sliding_window_size = 17
    num_heads = 8
    core_num_for_global = 3
    core_num_for_local = 2

    b = 2
    B = 3
    L = 190
    g2lmap = [0, 1, 1]
    global_x = torch.randn(b, L, in_dim)
    local_x = torch.randn(B, sliding_window_size, in_dim)
    obj = ModelCore(in_dim,
                    mid_dim,
                    out_dim,
                    sliding_window_size,
                    num_heads,
                    core_num_for_global,
                    core_num_for_local)
    o = obj(global_x, local_x, g2lmap)
    print(o.size())
    exit()
    """
    def __init__(
            self,
            in_dim,
            mid_dim,
            out_dim,
            sliding_window_size,
            num_heads,
            core_num_for_global,
            core_num_for_local
    ):
        super().__init__()
        self.globalnn = JGlobalSeqFea(in_dim, mid_dim, 128, sliding_window_size, num_heads, core_num_for_global)
        self.localnn = JLocalSeqFea(in_dim, mid_dim, 128, num_heads, core_num_for_local)

        self.mlp = nn.Sequential(
            nn.Linear(128*2*sliding_window_size, mid_dim),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 2 * mid_dim),
            nn.Linear(2 * mid_dim, out_dim)
        )

    @staticmethod
    def map_global_2_local(global_x, g2lmap):
        b, l, d = global_x.shape
        B = len(g2lmap)

        ans = torch.zeros(B, l, d).to(global_x.device)
        for ind in range(B):
            ans[ind] = global_x[g2lmap[ind], :, :]

        return ans

    def forward(
            self,
            global_x,
            local_x,
            g2lmap,
    ):
        """
        :param global_x: (tensor), [b, L, D]
        :param global_x: (tensor), [B, l, D]
        :param g2lmap: (list), [B], each element belongs to [1, 2, ..., b]
        :return:
        """
        global_x = self.globalnn(global_x)
        local_x = self.localnn(local_x)

        global_x = self.map_global_2_local(global_x, g2lmap)

        global_x = global_x.reshape(global_x.size(0), -1)
        local_x = local_x.reshape(local_x.size(0), -1)

        x = torch.cat((local_x, global_x), dim=-1)
        o = self.mlp(x)

        return o


class JModel(nn.Module):
    def __init__(self, core_model):
        super(JModel, self).__init__()
        self.core_model = core_model
        self.useless = nn.Parameter(torch.zeros(1))

    def save_core_model(self, savepath):
        checkpoint = {'model': self.core_model.state_dict()}
        torch.save(checkpoint, savepath)

    def forward(self, samples):
        device = self.useless.device

        global_x = samples['global_x'].to(device)
        local_x = samples['local_x'].to(device)
        g2lmap = samples['g2lmap']

        # {
        #     'global_x': global_x,  # tensor
        #     'local_x': local_x,  # tensor
        #     'g2lmap': g2lmap  # list
        # }

        core_model = self.core_model.to(device)
        return {
            'logit': core_model(
                global_x,
                local_x,
                g2lmap
            )
        }


def load_model():
    model = ModelCore(
        in_dim=5120,
        mid_dim=128,
        out_dim=2,
        sliding_window_size=17,
        num_heads=8,
        core_num_for_global=48,
        core_num_for_local=36
    )

    return model


def gen_input(esm2emb):
    """
    esm2emb: [L, D]
    """
    L, _ = esm2emb.size()

    prot_ind = 0
    g2lmap = []
    global_x = esm2emb.unsqueeze(0)
    local_x = []
    for ind in range(L):
        local_x.append(
            pick_up_one_sliding_window(esm2emb, ind, winsize=17).unsqueeze(0)
        )

        g2lmap.append(prot_ind)

    local_x = torch.cat(local_x, dim=0)

    return {
        'global_x': global_x,  # tensor
        'local_x': local_x,  # tensor
        'g2lmap': g2lmap  # list
    }


def parsePredProbs(outs):
    """
    :param outs [Tensor]: [*, 2 or 1]
    :return pred_probs: [*], tgts: [*]
    """

    # 1 : one probability of each sample
    # 2 : two probabilities of each sample
    __type = 1
    if outs.size(-1) == 2:
        __type = 2
        outs = outs.view(-1, 2)
    else:
        outs = outs.view(-1, 1)

    sam_num = outs.size(0)

    outs = outs.tolist()

    pred_probs = []
    for j in range(sam_num):
        out = outs[j]
        if 2 == __type:
            prob_posi = out[1]
            prob_nega = out[0]
        else:
            prob_posi = out[0]
            prob_nega = 1.0 - prob_posi

        sum = prob_posi + prob_nega

        if sum < 1e-99:
            pred_probs.append(0.)
        else:
            pred_probs.append(prob_posi / sum)

    return pred_probs


if __name__ == '__main__':

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-seq_fa", "--seq_fa")
    parser.add_argument("-sind", "--start_index", type=int, default=0)
    parser.add_argument("-eind", "--end_index", type=int, default=-1)
    parser.add_argument("-sf", "--savefolder", default="./")
    parser.add_argument("-dv", "--device", default='cuda:0')
    args = parser.parse_args()

    if args.seq_fa is None:
        parser.print_help()
        exit("PLEASE INPUT YOUR PARAMETERS CORRECTLY")

    print_namespace(args)
    seq_fa = args.seq_fa

    esm2m = "{}/model/esm2_t48_15B_UR50D.pt".format(os.path.abspath('.'))
    gle2eatpm = "{}/model/gle2eatpm.pkl".format(os.path.abspath('.'))
    print(esm2m)
    print(gle2eatpm)
    if not exists(esm2m) or not exists(gle2eatpm):
        exit("Please run the linux shell script of ./model/run.sh")

    device = args.device if torch.cuda.is_available() else 'cpu'
    feature = JSeq2ESM2(esm2_model_path=esm2m, device=device)

    checkpoint = torch.load(gle2eatpm, map_location=device, weights_only=False)
    state_dict = checkpoint['model']

    model = JModel(load_model()).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    seq_dict = loadFasta(seq_fa)

    start_index = args.start_index
    end_index = args.end_index
    if end_index <= start_index:
        end_index = len(seq_dict)

    keys = []
    for key in seq_dict:
        keys.append(key)

    createFolder(args.savefolder)

    tot_seq_num = len(seq_dict)
    for ind in range(tot_seq_num):
        if ind < start_index or ind >= end_index:
            continue
        key = keys[ind]
        seq = seq_dict[key]

        emb = feature.embed(seq)[0]
        samples = gen_input(emb)
        out = model(samples)['logit']

        out = torch.softmax(out, dim=-1)
        probs = parsePredProbs(out)

        filepath = "{}/{}.pred".format(args.savefolder, key)
        with open(filepath, 'w') as file_object:
            length = len(probs)
            file_object.write("Index    AA    Prob.    State\n")
            for i in range(length):
                aa = seq[i]
                prob = probs[i]
                file_object.write("{:5d}     {}    {:.8f}\n".format(i, aa, probs[i]))
            file_object.close()

        if ind % 1 == 0:
            print("The {}/{}-th {} with {} residues is predicted.".format(ind, tot_seq_num, key, len(seq)))

    print("Hope the predicted results could help you!")
