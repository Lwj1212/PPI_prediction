# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:32:21 2021

@author: zju
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from operator import itemgetter

import tensorflow as tf
import torch
from tqdm import tqdm

from Bio import SeqIO
from bio_embeddings.embed import BeplerEmbedder

def preprocess(pair_file, seq_file):
    
    with open(pair_file, 'r') as f:
        
        lines = f.readlines()
        
    proteins_1 = [line.strip().split('\t')[0] for line in lines]
    proteins_2 = [line.strip().split('\t')[1] for line in lines]
    labels = [line.strip().split('\t')[2] for line in lines]
    
    protein_list = list(set(proteins_1 + proteins_2))
            
    protein_seq = {}
           
    with open(seq_file, 'r') as f:
        
        lines = f.readlines()
        
    for i in range(len(lines)):
        
        line = lines[i].strip().split('\t')
        protein_seq[line[0]] = line[1]
        
#    return proteins_1, proteins_2, labels, protein_seq
    amino_acid ={'A':1,'C':2,'D':3,'E':4,'F':5,
                'G':6,'H':7,'I':8,'K':9,'L':10,
                'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
                'T':17,'V':18,'W':19,'Y':20,'U':21,'X':22,'B':0}

# positive and negative setting
    k1 = []
    k2 = []
    k3 = []
    k_h = []

    for i in range(len(labels)):

        protein_1 = proteins_1[i]
        protein_2 = proteins_2[i]
        
        label = labels[i]
        
        seq_1 = protein_seq[protein_1]
        seq_2 = protein_seq[protein_2]

        a1 = np.zeros([1500,], dtype = int)
        a2 = np.zeros([1500,], dtype = int)
        a3 = np.zeros([3,], dtype = float)
    
        k = 0
        for AA in seq_1:
            a1[k] = amino_acid[AA]
            k += 1
        k1.append(a1)
    
        k = 0
        for AA in seq_2:
            a2[k] = amino_acid[AA]
            k += 1
        k2.append(a2)
        
    
        if int(label) == 0:
            a3[1] = 1
        elif int(label) == 1:
            a3[0] = 1
        else:
            print('error')
            break
        k3.append(a3)
        
        k_h.append(np.array([protein_1, protein_2]))
    
    m1 = np.stack(k1, axis=0)
    m2 = np.stack(k2, axis=0)
    m3 = np.stack(k3, axis=0)
    m_h = np.stack(k_h, axis=0)

# single protein setting
    k1 = []
    k2 = []
    k3 = []

    for protein in protein_list:

        seq_1 = protein_seq[protein]
        seq_2 = 'B'
        label = 2
    
        a1 = np.zeros([1500,], dtype = int)
        a2 = np.zeros([1500,], dtype = int)
        a3 = np.zeros([3,], dtype = float)
    
        k = 0
        for AA in seq_1:
            a1[k] = amino_acid[AA]
            k += 1
        k1.append(a1)
    
        k = 0
        for AA in seq_2:
            a2[k] = amino_acid[AA]
            k += 1
        k2.append(a2)
        
        if int(label) == 2:
            a3[2] = 1
        else:
            print('error')
            break
        k3.append(a3)
    
    n1 = np.stack(k1, axis=0)
    n2 = np.stack(k2, axis=0)
    n3 = np.stack(k3, axis=0)
    
    return m1, m2, m3, n1, n2, n3, m_h

def embed_sequence(model, x, pool='none', use_cuda=False):
    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.cuda()

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = model.transform(x)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z

def preprocess_embed(id2seq_file, ds_file):
    id2index = {}
    seqs = []
    index = 0
    sid1_index = 0
    sid2_index = 1
    label_index = 2
    
    for line in open(id2seq_file):
        line = line.strip().split('\t')
        id2index[line[0]] = index
        seqs.append(line[1])
        index += 1

    seq_array = []
    id2_aid = {}
    sid = 0

    max_data = -1
    limit_data = max_data > 0
    raw_data = []
    skip_head = True
    x = None
    count = 0
    
    # Create sequence array as a list of protein strings
    for line in tqdm(open(ds_file)):
        if skip_head:
            skip_head = False
            continue
        line = line.rstrip('\n').rstrip('\r').split('\t')
        if id2index.get(line[sid1_index]) is None or id2index.get(line[sid2_index]) is None:
            continue
        if id2_aid.get(line[sid1_index]) is None:
            id2_aid[line[sid1_index]] = sid
            sid += 1
            seq_array.append(seqs[id2index[line[sid1_index]]])
        line[sid1_index] = id2_aid[line[sid1_index]]
        if id2_aid.get(line[sid2_index]) is None:
            id2_aid[line[sid2_index]] = sid
            sid += 1
            seq_array.append(seqs[id2index[line[sid2_index]]])
        line[sid2_index] = id2_aid[line[sid2_index]]
        raw_data.append(line)
        if limit_data:
            count += 1
            if count >= max_data:
                break

    # Extract index of 1st and 2nd sequences in pairs
    seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
    seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

    # Assign labels for pairs of sequences
    class_map = {'0': 1, '1': 0}
    class_labels = np.zeros((len(raw_data), 2))
    for i in range(len(raw_data)):
        class_labels[i][class_map[raw_data[i][label_index]]] = 1
    

    # Pretrained embed
    embedder = BeplerEmbedder()
    sequences = pd.read_csv(id2seq_file, sep="\t", header=None)
    sequences = sequences.iloc[:,1].to_list()
        
    embeddings = []
    i = 1
    for sequence in sequences:
        embeddings.append(embedder.embed(sequence))
        if i % 1000 == 0:
            print(i)
        i+=1
        
    embeddings = list(embeddings)

    seq_tensor= tf.keras.preprocessing.sequence.pad_sequences(embeddings,  padding='post', dtype='float16', truncating='post', maxlen=2000)
    dim = seq_tensor.shape[2]
    
    Path('preprocess').mkdir(parents=True, exist_ok=True)
    np.savez('preprocess/embed_preprocess.npz', seq_tensor = seq_tensor, seq_index1 = seq_index1, 
                                            seq_index2 = seq_index2, class_labels = class_labels, dim = dim)
    

    return seq_tensor, seq_index1, seq_index2, class_labels, dim