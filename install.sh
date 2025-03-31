#!/bin/bash

if [ ! -f "./model/esm2_t48_15B_UR50D.pt" ]; then
	wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt -O ./model/esm2_t48_15B_UR50D.pt
fi

if [ ! -f "./model/esm2_t48_15B_UR50D-contact-regression.pt" ]; then
	wget -c https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t48_15B_UR50D-contact-regression.pt -O ./model/esm2_t48_15B_UR50D-contact-regression.pt
fi

if [ ! -f "./model/gle2eatpm.pkl" ]; then
	java -jar FileUnion.jar ./model/gle2eatpm ./model/gle2eatpm.pkl
fi

echo "Installed."
