#!/bin/bash

MUSDB18hq="/home/sahand/BandSplit-RNN/musdb18hq/"

python3 ./bsrnn.py --musdbhq_location "$MUSDB18hq" --mode "train-base-model" --batch_size 2