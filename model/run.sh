#!/bin/bash

wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt -O esm2_t48_15B_UR50D.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t48_15B_UR50D-contact-regression.pt -O esm2_t48_15B_UR50D-contact-regression.pt

java -jar FileUnion.jar ./gle2eatpm ./gle2eatpm.pkl
