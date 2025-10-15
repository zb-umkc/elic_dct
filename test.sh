#!/bin/bash

for grps in "1234"; do
    echo "Bypass Groups: ${grps}"
    for lmbda in 1; do
        echo "-- lambda=${lmbda}"
        python3 test_NGA_DCTv2_RLE_grps.py --mode test --primary_pol HH --test-dataset /scratch/zb7df/data/NGA/multi_pol/validation --test-model /scratch/zb7df/data/checkpoint-NGA-DCTv2/checkpoint_best_lambda${lmbda}.pth.tar --bypass-grps $grps
    done
done