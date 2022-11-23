#!/bin/bash

# arg
fasta_path=${1}   # require gzip
out_path=${2}

gzip -f -k ${fasta_path}

# conda env
source /home/wmbio/anaconda3/etc/profile.d/conda.sh
conda activate deeptrio-gpu

# run cd-hit
cd-hit -i ${fasta_path} -o ${out_path} -n 2 -c 0.4 -T 15 -M 1000

