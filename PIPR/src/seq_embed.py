import tensorflow as tf

from bio_embeddings.embed import BeplerEmbedder,ProtTransT5XLU50Embedder
# Import accessory modules
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse
import os

def seq_max(id2seq_file):
    seqs = []
    for line in open(id2seq_file):
        line = line.strip().split('\t')
        seqs.append(len(line[1]))
    
    return max(seqs)

def preprocess_embed(id2seq_file, ds_file, e_type):
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
    if e_type == "bepler":
      embedder = BeplerEmbedder()
    elif e_type == "prottrans_t5u50":
      embedder = ProtTransT5XLU50Embedder()
        
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
                                                              dtype='float16', truncating='post', maxlen=seq_max(id2seq_file))
    dim = seq_tensor.shape[2]
    
    Path('preprocess').mkdir(parents=True, exist_ok=True)
    np.savez('preprocess/embed_preprocess_' + e_type + '.npz', 
             seq_tensor = seq_tensor, seq_index1 = seq_index1, 
             seq_index2 = seq_index2, class_labels = class_labels, dim = dim)
    

    return seq_tensor, seq_index1, seq_index2, class_labels, dim

if __name__ == "__main__":
  # ============================================
  # Optimisation Flags - Do not remove
  # ============================================

  ### Setting RAM GPU for training growth 
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
      try:
          # Currently, memory growth needs to be the same across GPUs
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
          print(e)

  # Disables caching (when set to 1) or enables caching (when set to 0) for just-in-time-compilation. When disabled,
  # no binary code is added to or retrieved from the cache.
  os.environ['CUDA_CACHE_DISABLE'] = '0' # orig is 0

  # When set to 1, forces the device driver to ignore any binary code embedded in an application 
  # (see Application Compatibility) and to just-in-time compile embedded PTX code instead.
  # If a kernel does not have embedded PTX code, it will fail to load. This environment variable can be used to
  # validate that PTX code is embedded in an application and that its just-in-time compilation works as expected to guarantee application 
  # forward compatibility with future architectures.
  os.environ['CUDA_FORCE_PTX_JIT'] = '1'# no orig
  os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
  os.environ['TF_GPU_THREAD_COUNT']='1'
  os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  os.environ['TF_ADJUST_HUE_FUSED'] = '1'
  os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
  os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
  os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  # =================================================
  # mixed_precision.set_global_policy('mixed_float16')

  # argpaser
  parser = argparse.ArgumentParser(description='pLM Embedding & Input preprocessing')
  parser.add_argument('-p', '--ppi', required=True, type=str, help='configuration of the PPI file, which contains the protein 1 id, protein 2 id and the class they belong to, and they are splited by table key')
  parser.add_argument('-d', '--database', required=True, type=str, help='configuration of the protein sequence database, which contains the protein id and its sequence, and they are splited by table key')
  parser.add_argument('-e', '--etype', required=True, type=str, help='bepler or prottrans_t5u50')

  static_args = parser.parse_args()
  file_1_path = static_args.ppi
  file_2_path = static_args.database
  EMBEDDING_TYPE = static_args.etype
  DTYPE='float16'

  print('run embed!')
  print(EMBEDDING_TYPE)
  preprocess_embed(id2seq_file=file_2_path, ds_file=file_1_path, e_type=EMBEDDING_TYPE)
