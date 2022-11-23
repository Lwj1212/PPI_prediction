# Libraries for system and debug
from ast import Param
import sys
import pdb
import os
from datetime import datetime

# Class for converting sequences to tensors
from seq2tensor import s2t

# Libraries for neural network training
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional, Input, Conv1D, Conv2D
from tensorflow.keras.layers import Add, Flatten, subtract, multiply, concatenate
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.utils import Sequence
from tensorflow.keras import mixed_precision
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import train_test_split


# Import accessory modules
import numpy as np
import h5py
import gc
from tqdm import tqdm


# Function
def generator_pair(seq_tensor, class_labels, pair_index):
  for index in pair_index:
    yield {"seq1": seq_tensor[seq_index1[index]], "seq2": seq_tensor[seq_index2[index]]}, class_labels[index]


def generator_pair_predict(seq_tensor, class_labels, pair_index): 
  for index in pair_index:
    yield {"seq1": seq_tensor[seq_index1[index]], "seq2": seq_tensor[seq_index2[index]]}


def input_preprocess(id2seq_file, ds_file, use_emb):
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

  seq2t = s2t(use_emb)
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

    len_m_seq = np.array([len(line.split()) for line in seq_array])
    avg_m_seq = int(np.average(len_m_seq)) + 1
    max_m_seq = max(len_m_seq)
    dim = seq2t.dim

    # seq_tensor is tensor representation of dataset having shape of (number_of_sequences, padding_length, embedding_dim_of_aa)
    # Random for distribution of class labels
    seq_tensor = np.array([seq2t.embed_normalized(line, seq_size)
                          for line in tqdm(seq_array)]).astype('float16')

    # Extract index of 1st and 2nd sequences in pairs
    seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
    seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

    # Assign labels for pairs of sequences
    class_map = {'0': 1, '1': 0}
    class_labels = np.zeros((len(raw_data), 2))
    for i in range(len(raw_data)):
        class_labels[i][class_map[raw_data[i][label_index]]] = 1

    return seq_tensor, seq_index1, seq_index2, class_labels


def build_model(hparams):
  # Input of sequence tensor representations
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')

    # Define Conv1D and Bi-RNN (GRU/LSTM) use in architecture
    l1 = Conv1D(hparams[HP_CONV_HIDDEN_DIM], hparams[HP_KERNEL_SIZE],
                activation=hparams[HP_ACTIVATION_CONV], padding=hparams[HP_CONV_PADDING])
    r1 = Bidirectional(GRU(hparams[HP_RNN_HIDDEN_DIM], return_sequences=True))
    l2 = Conv1D(hparams[HP_CONV_HIDDEN_DIM], hparams[HP_KERNEL_SIZE],
                activation=hparams[HP_ACTIVATION_CONV], padding=hparams[HP_CONV_PADDING])
    r2 = Bidirectional(GRU(hparams[HP_RNN_HIDDEN_DIM], return_sequences=True))
    l3 = Conv1D(hparams[HP_CONV_HIDDEN_DIM], hparams[HP_KERNEL_SIZE],
                activation=hparams[HP_ACTIVATION_CONV], padding=hparams[HP_CONV_PADDING])
    r3 = Bidirectional(GRU(hparams[HP_RNN_HIDDEN_DIM], return_sequences=True))
    l4 = Conv1D(hparams[HP_CONV_HIDDEN_DIM], hparams[HP_KERNEL_SIZE],
                activation=hparams[HP_ACTIVATION_CONV], padding=hparams[HP_CONV_PADDING])
    r4 = Bidirectional(GRU(hparams[HP_RNN_HIDDEN_DIM], return_sequences=True))
    l5 = Conv1D(hparams[HP_CONV_HIDDEN_DIM], hparams[HP_KERNEL_SIZE],
                activation=hparams[HP_ACTIVATION_CONV], padding=hparams[HP_CONV_PADDING])
    r5 = Bidirectional(GRU(hparams[HP_RNN_HIDDEN_DIM], return_sequences=True))
    l6 = Conv1D(hparams[HP_CONV_HIDDEN_DIM], hparams[HP_KERNEL_SIZE],
                activation=hparams[HP_ACTIVATION_CONV], padding=hparams[HP_CONV_PADDING])

    # Siamese architecture

    ### 1st sibling

    # 1st Block RCNN
    s1 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l1(seq_input1))
    s1 = concatenate([r1(s1), s1])

    # 2nd Block RCNN
    s1 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l2(s1))
    s1 = concatenate([r2(s1), s1])

    # 3rd Block RCNN
    s1 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l3(s1))
    s1 = concatenate([r3(s1), s1])

    # 4th Block RCNN
    s1 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l4(s1))
    s1 = concatenate([r4(s1), s1])

    # 5th Block RCNN
    s1 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l5(s1))
    s1 = concatenate([r5(s1), s1])

    # Last convolution
    s1 = l6(s1)
    s1 = GlobalAveragePooling1D()(s1)

    ### 2nd sibling

    # 1st block RCNN
    s2 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l1(seq_input2))
    s2 = concatenate([r1(s2), s2])

    # 2nd block RCNN
    s2 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l2(s2))
    s2 = concatenate([r2(s2), s2])

    # 3rd block RCNN
    s2 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l3(s2))
    s2 = concatenate([r3(s2), s2])

    # 4th block RCNN
    s2 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l4(s2))
    s2 = concatenate([r4(s2), s2])

    # 5th block RCNN
    s2 = MaxPooling1D(hparams[HP_POOLING_KERNEL])(l5(s2))
    s2 = concatenate([r5(s2), s2])

    # Last convolution
    s2 = l6(s2)
    s2 = GlobalAveragePooling1D()(s2)

    ### Combine two siblings of siamese architecture
    merge_text = multiply([s1, s2])

    #### MLP Part
    # Set initializer

    # First dense
    x = Dense(hparams[HP_FIRST_DENSE],
              activation=hparams[HP_ACTIVATION])(merge_text)
    # x = tf.keras.layers.LeakyReLU(alpha=.3)(x)
    x = Dropout(hparams[HP_DROPOUT])(x)

    # Second dense
    x = Dense(int((hparams[HP_CONV_HIDDEN_DIM]+7)/2),
              activation=hparams[HP_ACTIVATION])(x)
    # x = tf.keras.layers.LeakyReLU(alpha=.3)(x)
    x = Dropout(hparams[HP_DROPOUT])(x)

    # Last softmax
    main_output = Dense(2, activation='softmax')(x)

    # Combine to form functional model
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model


if __name__ == "main":
  # ============================================
  # Optimisation Flags - Do not remove
  # ============================================

  # Disables caching (when set to 1) or enables caching (when set to 0) for just-in-time-compilation. When disabled,
  # no binary code is added to or retrieved from the cache.
  os.environ['CUDA_CACHE_DISABLE'] = '0'  # orig is 0

  # When set to 1, forces the device driver to ignore any binary code embedded in an application
  # (see Application Compatibility) and to just-in-time compile embedded PTX code instead.
  # If a kernel does not have embedded PTX code, it will fail to load. This environment variable can be used to
  # validate that PTX code is embedded in an application and that its just-in-time compilation works as expected to guarantee application
  # forward compatibility with future architectures.
  os.environ['CUDA_FORCE_PTX_JIT'] = '1'  # no orig

  os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
  os.environ['TF_GPU_THREAD_COUNT'] = '1'

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
