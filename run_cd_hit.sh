#!/bin/bash

# arg
fasta_path=${1}   # require gzip
out_path=${2}

gzip -f -k ${fasta_path}

# conda env
source /home/wmbio/anaconda3/etc/profile.d/conda.sh
conda activate deeptrio-gpu

# run cd-hit
# cd-hit -i ${fasta_path} -o ${out_path} -n 3 -c 0.6 -T 15 -M 1000
# cd-hit -i ${fasta_path} -o ${out_path} -n 4 -c 0.7 -T 15 -M 1000
cd-hit -i ${fasta_path} -o ${out_path} -n 5 -c 0.8 -T 15 -M 1000