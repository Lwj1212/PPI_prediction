'''

python src/prediction.py -b final_model/ -t baseline -d data/wmbio_set/Test_set/human_test_seq.tsv -p data/wmbio_set/Test_set/human_test_pair.tsv

python src/prediction.py -b final_model/ -t bepler -d data/wmbio_set/Test_set/human_test_seq.tsv -p data/wmbio_set/Test_set/human_test_pair.tsv -l 20220315-090206
'''

# Libraries for system and debug
import os
from datetime import datetime
from os import sep

# Libraries for neural network training
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import tensorflow_addons as tfa

# from bio_embeddings.embed import BeplerEmbedder,ProtTransT5XLU50Embedder,ESM1bEmbedder
from seq2tensor import s2t

# Import accessory modules
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse

def preprocess_prediction_embed(id2seq_file, ds_file, e_type, use_emb):
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
    x = None
    count = 0
    
    # Create sequence array as a list of protein strings
    for line in tqdm(open(ds_file)):
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
    seq_size = 1499
    
    if e_type == "baseline":
        seq2t = s2t(use_emb)
        seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)]).astype('float32')
    else :
        # Pretrained embed
        if e_type == "bepler":
            embedder = BeplerEmbedder()
        elif e_type == "prottrans_t5u50":
            embedder = ProtTransT5XLU50Embedder()

        # seq
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

        seq_tensor= tf.keras.preprocessing.sequence.pad_sequences(embeddings,  padding='post', 
                                                                  dtype='float32', truncating='post', maxlen=seq_size)     
    # pair
    pairs = pd.read_csv(ds_file, sep="\t", header=None)
    pairs = list(zip(pairs.iloc[:, 0], pairs.iloc[:, 1]))
    pairs = list(map(lambda x : "-".join(x), pairs))
    
    # save np
    Path('preprocess_predict').mkdir(parents=True, exist_ok=True)
    FILE_NAME = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.savez('preprocess_predict/' + FILE_NAME + '.npz', 
             seq_tensor = seq_tensor, seq_index1 = seq_index1, 
             seq_index2 = seq_index2, pairs = pairs)

    return seq_tensor, seq_index1, seq_index2, pairs

def seq_max(id2seq_file):
    seqs = []
    for line in open(id2seq_file):
        line = line.strip().split('\t')
        seqs.append(len(line[1]))
    
    return max(seqs)


def pipr_prediction(model_path, model_type, id2seq_file, ds_file, use_emb=None, load=None):
    '''
    model_type : bepler, prottrans_t5u50, baseline
    '''
    if load == None:
        seq_tensor, seq_index1, seq_index2, pairs = preprocess_prediction_embed(id2seq_file=id2seq_file,
                                                                             ds_file=ds_file,
                                                                             e_type=model_type,
                                                                             use_emb=use_emb)
    else :
        with np.load('preprocess_predict/' + load + '.npz') as data:
            seq_tensor, seq_index1, seq_index2, pairs = data['seq_tensor'], data['seq_index1'], data['seq_index2'], data['pairs']

    with tf.device('/CPU:0'):
      model = tf.keras.models.load_model(model_path + 'PIPR_' + model_type + '_final.h5')
      prediction = pd.DataFrame(model.predict_on_batch([seq_tensor[seq_index1], seq_tensor[seq_index2]]), 
                                columns=["True", "False"], index = pairs)
      
    return prediction                        



if __name__ == "__main__":
  # argpaser
  parser = argparse.ArgumentParser(description='PIPR prediction')
  parser.add_argument('-b', '--path', default='final_model/', type=str, help='model path')
  parser.add_argument('-t', '--type', default="baseline", type=str, help='model type : bepler, prottrans_t5u50, baseline')
  parser.add_argument('-d', '--database', required=True, type=str, help='configuration of the protein sequence database, which contains the protein id and its sequence, and they are splited by table key')
  parser.add_argument('-p', '--ppi', required=True, type=str, help='configuration of the PPI file, which contains the protein 1 id, protein 2 id and the class they belong to, and they are splited by table key')
  parser.add_argument('-l', '--load', default=None, type=str, help='preprocessing npz file')
  parser.add_argument('-e', '--embed', default="final_model/ac5_aph.txt", type=str, help='baseline embedder')
  parser.add_argument('-o', '--out', default=os.getcwd(), type=str, help='save path')
  
  static_args = parser.parse_args()
  model_path = static_args.path
  model_type = static_args.type
  id2seq_file = static_args.database
  ds_file = static_args.ppi
  load = static_args.load
  use_emb = static_args.embed
  out = static_args.out

  # model_type : bepler, prottrans_t5u50, baseline
  result = pipr_prediction(model_path=model_path, 
                  model_type=model_type, 
                  id2seq_file=id2seq_file,
                  ds_file=ds_file,
                  use_emb=use_emb,
                  load=load
                  )

  FILE_NAME = datetime.now().strftime("%Y%m%d-%H%M%S")
  # Path(out + '/prediction_result').mkdir(parents=True, exist_ok=True)
  result.to_csv(out + '/' + FILE_NAME + '_predictions.tsv', sep="\t")
  # print(result)
  