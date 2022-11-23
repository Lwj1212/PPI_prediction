#!/bin/bash

while getopts b:r:d:m: flag
do
    case "${flag}" in
        b) base=${OPTARG};;
        r) result=${OPTARG};;
        d) device=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

# conda env activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ppi_prediction

# result directory
mkdir -p ${base}/embedding
mkdir -p ${result}

# embed & predict
for embed in ${base}/*.fasta
do  
    filename=$(basename ${embed} .fasta)
    echo ${filename}

    # embed
    dscript embed --seqs ${base}/${filename}.fasta --outfile ${base}/embedding/${filename}.h5 -d ${device}

    # predict
    dscript predict --pairs ${base}/${filename}.tsv --embeddings ${base}/embedding/${filename}.h5 -o ${result}/${filename} --model ${model} -d ${device}
done
