# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:24:08 2019

example : python src/build_model_for_hyperparameter_search_pretrained.py \
    -p /home/wmbio/WORK/gitworking/D-SCRIPT/data/wmbio_set/train/human_custom_toy.tsv \
    -d /home/wmbio/WORK/gitworking/D-SCRIPT/data/wmbio_set/train/human_custom_toy.fasta \
    -e 50 -b 32

@author: zju
@convert: jinoo
"""
import os
import gc
import datetime
from math import log
from pathlib import Path
import numpy as np
import pandas as pd
from operator import itemgetter

import torch
from prose.alphabets import Uniprot21
from prose.models.multitask import ProSEMT
import prose.fasta as fasta

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa

from sklearn.model_selection import KFold, ShuffleSplit
from build_my_layer import MyMaskCompute, MySpatialDropout1D
from utility import array_split_prose, random_arr, array_split
from input_preprocess import preprocess, preprocess_embed 
import warnings
import argparse
import GPy
import GPyOpt


# function
def main(sp_drop=0.005, kernel_rate_1=0.16, strides_rate_1=0.15, kernel_rate_2=0.14, strides_rate_2=0.25, 
        filter_num_1=150, filter_num_2=175, con_drop=0.05, fn_drop_1=0.2, fn_drop_2=0.1, node_num=256, opti_switch=1):
    
    if opti_switch == 0:
        adam = Adam(amsgrad = False)
        # print('^^^^^ False ^^^^^')
    elif opti_switch == 1:
        adam = Adam(amsgrad = True)
        # print('^^^^^ True ^^^^^')
    else:
        raise Exception('The format is not in a right way')
    
    main_input_a = Input(shape = (seq_size, dim), name = 'seqA')
    main_input_b = Input(shape = (seq_size, dim), name = 'seqB')

    masked_a = MyMaskCompute()(main_input_a)
    masked_b = MyMaskCompute()(main_input_b)

    drop_layer = MySpatialDropout1D(sp_drop)

    dropped_1 = drop_layer(masked_a)
    dropped_2 = drop_layer(masked_b)

    tensor = []

    for n in range(2,35):
        
        if n <= 15:
            conv_layer = Conv1D(filters= int(filter_num_1),
            kernel_size = int(np.ceil(kernel_rate_1 * n**2)),
            padding = 'valid',
            activation = 'relu',
            use_bias= False,
            strides = int(np.ceil(strides_rate_1*(n-1))))
        else:
            conv_layer = Conv1D(filters= int(filter_num_2),
            kernel_size = int(np.ceil(kernel_rate_2 * n**2)),
            padding = 'valid',
            activation = 'relu',
            use_bias= False,
            strides = int(np.ceil(strides_rate_2*(n-1))))
        
        conv_out_1 = conv_layer(dropped_1)
        conv_out_2 = conv_layer(dropped_2)

        conv_out_1 = SpatialDropout1D(con_drop)(conv_out_1)
        conv_out_2 = SpatialDropout1D(con_drop)(conv_out_2)
        
        max_layer = MaxPooling1D(pool_size=int(conv_layer.output_shape[1]))
        
        pool_out_1 = max_layer(conv_out_1)
        pool_out_2 = max_layer(conv_out_2)
        
        pool_out = pool_out_1 + pool_out_2
        
        flat_out = Flatten()(pool_out)
        
        tensor.append(flat_out)
        
    concatenated = Concatenate()(tensor)
    x = Dropout(fn_drop_1)(concatenated)
    x = Dense(int(node_num))(x)
    x = Dropout(fn_drop_2)(x)
    x = Activation('relu')(x)
    main_output = Dense(2, activation='softmax', name = 'out')(x)  # sigle protein prediction removal
  
    # model fit
    model = Model(inputs = [main_input_a,main_input_b], outputs = main_output)

    # accuracy
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    # f1-score
    # model.compile(loss='categorical_crossentropy', optimizer=adam, 
    #               metrics=[tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.5)])
    record_min = 0

    # tensorbaord
    TB_log_dir = make_Tensorboard_dir("Learning_log")
    # callbacks
    callbacks = [
                    EarlyStopping(monitor='val_loss', patience=4, verbose=1),
                    ModelCheckpoint(filepath = sub_dir_name + '/DeepTrio_search_' + str(yy) + '.h5',
                                 monitor='val_loss', save_best_only=True),
                    TensorBoard(log_dir = TB_log_dir)
                ]

    history_model = model.fit(train_dataset, 
                              batch_size=batch_size, 
                              epochs=epoch_number, 
                              shuffle=True,    
                              callbacks=callbacks,
                              validation_data=test_dataset)
    
    record_min = history_model.history['val_f1_score'][0]

    del history_model
    del model

    K.clear_session()
    gc.collect()

    with open(sub_dir_name + '/search_log.txt', 'a') as log_text:
                
        log_text.write('cycle: ' + str(yy) + '\n')
        log_text.write('sp_drop: ' + str(sp_drop) + '\n')
        log_text.write('kernel_rate_1: ' + str(kernel_rate_1) + '\n')
        log_text.write('strides_rate_1: ' + str(strides_rate_1) + '\n')
        log_text.write('kernel_rate_2: ' + str(kernel_rate_2) + '\n')
        log_text.write('strides_rate_2: ' + str(strides_rate_2) + '\n')
        log_text.write('filter_num_1: ' + str(filter_num_1) + '\n')
        log_text.write('filter_num_2: ' + str(filter_num_2) + '\n')
        log_text.write('con_drop: ' + str(con_drop) + '\n')
        log_text.write('fn_drop_1: ' + str(fn_drop_1) + '\n')
        log_text.write('fn_drop_2: ' + str(fn_drop_2) + '\n')
        log_text.write('node_num: ' + str(node_num) + '\n')
        log_text.write('opti_switch: ' + str(opti_switch) + '\n')

        log_text.write('accuracy: ' + str(record_min) + '\n')
        # log_text.write('f1_score: ' + str(record_min) + '\n')
        log_text.write('-----\n')

    return record_min

bounds = [
          #Discrete 
          {'name':'sp_drop','type':'discrete','domain':(0.005, 0.01)},
          {'name':'kernel_rate_1','type':'discrete','domain':(0.12, 0.14, 0.16)},
          {'name':'strides_rate_1','type':'discrete','domain':(0.15, 0.2, 0.25)},
          {'name':'kernel_rate_2','type':'discrete','domain':(0.1, 0.12, 0.14)},
          {'name':'strides_rate_2','type':'discrete','domain':(0.25, 0.3, 0.35)},
          {'name':'filter_num_1','type':'discrete','domain':(100, 125, 150)},
          {'name':'filter_num_2','type':'discrete','domain':(150, 175)},
          {'name':'con_drop', 'type': 'discrete','domain': (0.05, 0.1, 0.15)},
          {'name':'fn_drop_1', 'type': 'discrete','domain': (0.1, 0.2)},
          {'name':'fn_drop_2', 'type': 'discrete','domain': (0.1, 0.2)},
          {'name':'node_num', 'type': 'discrete','domain': (128, 256)},
          #Categorical
          {'name':'opti_switch', 'type': 'categorical','domain': (0, 1)}
         ]

from dotmap import DotMap

# gpyopt function
def search_param(x):

    opt= DotMap()

    opt.em_dim = float(x[:, 0])
    opt.sp_drop = float(x[:, 1])
    opt.kernel_rate_1 = float(x[:, 2])
    opt.strides_rate_1 = float(x[:, 3])
    opt.kernel_rate_2 = float(x[:, 4])
    opt.strides_rate_2 = float(x[:, 5])
    opt.filter_num_1 = float(x[:, 6])
    opt.filter_num_2 = float(x[:, 7])
    opt.con_drop = float(x[:, 8])
    opt.fn_drop_1 = float(x[:, 9])
    opt.fn_drop_2 = float(x[:, 10])
    opt.node_num = float(x[:, 11])
    opt.opti_switch = int(x[:,12])

    return opt

yy = 0
def f(x):
    global yy
    
    local_yy = yy
    local_yy += 1
    yy = local_yy

    opt = search_param(x)
    param = {
            'sp_drop':opt.sp_drop,
            'kernel_rate_1':opt.kernel_rate_1,
            'strides_rate_1':opt.strides_rate_1,
            'kernel_rate_2':opt.kernel_rate_2,
            'strides_rate_2':opt.strides_rate_2,
            'filter_num_1':opt.filter_num_1,
            'filter_num_2':opt.filter_num_2,
            'con_drop':opt.con_drop,
            'fn_drop_1':opt.fn_drop_1,
            'fn_drop_2':opt.fn_drop_2,
            'node_num':opt.node_num,
            'opti_switch':opt.opti_switch
            }


    result = main(**param)

    evaluation = 1 - result

    # with open('search_log.txt', 'a') as log_text:
    #     log_text.write('evaluation: ' + str(evaluation) + '\n')
    #     log_text.write('---------\n')

    print('cycle: ' + str(yy))
    print('evaluation: ' + str(evaluation))

    return evaluation

def make_Tensorboard_dir(dir_name): 
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)

def generator_pair(seq_tensor, class_labels, pair_index):
    for index in pair_index:
        yield {"seq1": seq_tensor[seq_index1[index]], "seq2": seq_tensor[seq_index2[index]]}, class_labels[index]

def generator_pair_predict(seq_tensor, class_labels, pair_index):
    for index in pair_index:
        yield {"seq1": seq_tensor[seq_index1[index]], "seq2": seq_tensor[seq_index2[index]]}



if __name__ == "__main__": 

    # tensorflw configure
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'    

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8*1024)])

    # argpaser
    parser = argparse.ArgumentParser(description='train DeepTrio')
    parser.add_argument('-p', '--ppi', required=True, type=str, help='configuration of the PPI file, which contains the protein 1 id, protein 2 id and the class they belong to, and they are splited by table key')
    parser.add_argument('-d', '--database', required=True, type=str, help='configuration of the protein sequence database, which contains the protein id and its sequence, and they are splited by table key')
    parser.add_argument('-e', '--epoch', default='100', type=str, help='the maximum number of epochs')
    parser.add_argument('-b', '--batch', default=os.getcwd(), type=str, help='number of Batch size')

    static_args = parser.parse_args()
    file_1_path = static_args.ppi
    file_2_path = static_args.database
    epoch_number = int(static_args.epoch)
    batch_size = int(static_args.batch)
    DTYPE='float16'

    print('\nWelcome to use our tool')
    print('\nVersion: 1.0.0')
    print('\nAny problem, please contact mchen@zju.edu.cn')
    print('\nStart to process the raw data')

    # preprocess
    if os.path.isfile('preprocess/embed_preprocess.npz'):
      with np.load('preprocess/embed_preprocess.npz') as data:
        seq_tensor, seq_index1, seq_index2, class_labels, dim = data['seq_tensor'], data['seq_index1'], data['seq_index2'], data['class_labels'], data['dim']
    else :
      seq_tensor, seq_index1, seq_index2, class_labels, dim = preprocess_embed(file_1_path, file_2_path)

    seq_size = seq_tensor.shape[1]

    # train test split    ###
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train,test = list(kf.split(class_labels))[0]

    # from generator
    train_dataset = tf.data.Dataset.from_generator(generator_pair, 
                                                    args=[seq_tensor, class_labels, train], 
                                                    output_types=({"seq1": DTYPE, "seq2": DTYPE}, DTYPE), 
                                                    output_shapes = ({"seq1": (seq_size, dim), "seq2": (seq_size, dim)}, (2,)) )
    train_dataset = train_dataset.shuffle(1024).repeat(epoch_number).batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    # Create test
    test_dataset = tf.data.Dataset.from_generator(generator_pair, args=[seq_tensor, class_labels, test], 
                                                  output_types=({"seq1": DTYPE, "seq2": DTYPE}, DTYPE), 
                                                  output_shapes = ({"seq1": (seq_size, dim), "seq2": (seq_size, dim)}, (2,)) )
    test_dataset = test_dataset.batch(batch_size)  

    # save model dir
    sub_dir_name = "save_model_embed/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # save model parent dir
    Path(sub_dir_name).mkdir(parents=True, exist_ok=True)


    # Gpyopt
    print('\nStart to train DeepTrio model')
    print('\nAfter training, you may select the best model manually according to the recording file')
    opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=10)
    opt_model.run_optimization(max_iter=50)

    with open(sub_dir_name + '/search_log.txt', 'a') as log_text:
        log_text.write('result: \n')

        for i,v in enumerate(bounds):
            name = v['name']
            log_text.write('parameter {}: {}\n'.format(name,opt_model.x_opt[i]))
        log_text.write('evaluation: ' + str(1 - opt_model.fx_opt) + '\n')

    print('Congratulations, the training is complete')
