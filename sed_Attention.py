from __future__ import print_function
import os
import numpy as np
import time
import sys
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plot
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Permute, Merge,merge, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
import metrics
import utils
from IPython import embed
from keras.backend.tensorflow_backend import set_session
import scipy.io as sio
from attention_utils import get_activations

K.set_image_data_format('channels_first')
plot.switch_backend('agg')
sys.setrecursionlimit(10000)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

GPU = "1"
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

def GetEvents(pred, Y_test, thresh):
    pred = pred.reshape(pred.shape[0]*pred.shape[1],6)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if(pred[i,j] >= thresh):
                pred[i,j] = 1
            if(pred[i,j] < thresh):
                pred[i,j] = 0
    tar  = Y_test.reshape(Y_test.shape[0]*Y_test.shape[1],6)
    cnt = 0
    for i in range (tar.shape[0]):
        if((np.sum(tar[i,:]) < 2 )):
            cnt = cnt + 1
    p = np.zeros((cnt))
    t = np.zeros((cnt))
    cnt = 0
    for i in range (tar.shape[0]):
        if((np.sum(tar[i,:]) < 2 )):
            for j in range(6):
                if(tar[i,j] == 1):
                    t[cnt] = j + 1
                    break
            for j in range(6):
                if(pred[i,j] == 1):
                    p[cnt] = j + 1
                if(t[cnt] == j + 1):
                   break                                
            cnt = cnt + 1 
    return p,t 
def load_data(_feat_folder, _mono, _fold=None):
    feat_file_fold = _feat_folder + 'mbe_mon_fold0' + '.npz'
    dmp = np.load(feat_file_fold)
    _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
    return _X_train, _Y_train, _X_test, _Y_test


def get_model(data_in, data_out, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):

    spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))

    spec_x = spec_start
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((data_in.shape[-2], -1),name='conv_vec')(spec_x)


    spec_x_ = Permute((2,1))(spec_x)

    ### add the space attention model below 
    print(spec_x.shape)   
    attention_probs = Dense(256, activation='softmax', name='attention_vec')(spec_x_)
    #attention_probs = Dense(256, activation='softmax', name='attention_vec')(spec_x)
    print(attention_probs.shape)
    attention_probs = BatchNormalization()(attention_probs)
    attention_probs = Dropout(0.5)(attention_probs)
    attention_mul = merge([spec_x, attention_probs], name='attention_mul', mode='mul')   
    spec_x = Reshape((data_in.shape[-2], -1))(attention_mul)
    ### add the space attention model above

    for _r in _rnn_nb:
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul')(spec_x)

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = Model(inputs=spec_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy')
    _model.summary()
    return _model


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, _f1_t,_er_t):
    plot.figure()

    plot.subplot(311)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(_nb_epoch), _f1, label='f_validatation')
    plot.plot(range(_nb_epoch), _er, label='er_validation')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(_nb_epoch), _f1_t, label='f_test')
    plot.plot(range(_nb_epoch), _er_t, label='er_test')
    plot.legend()
    plot.grid(True)
    plotname = '/data/users/21799506/Data/PRL2018/Evaluation/models_Temp/performance.png'
    plot.savefig(plotname)
    plot.close()


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    print(_X.shape)
    print(_Y.shape)
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)
    print(_X.shape)
    print(_Y.shape)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test


#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################

is_mono = True  # True: mono-channel input, False: binaural input

feat_folder = '/data/users/21799506/Data/PRL2018/Evaluation/feat/'
__fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))


nb_ch = 1 if is_mono else 2
batch_size = 16    # Decrease this if you want to run on smaller GPU's
seq_len = 256       # Frame sequence length. Input to the CRNN.
nb_epoch = 150      # Training epochs
patience = int(0.2 * nb_epoch)  # Patience for early stopping

# Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
# Make sure the nfft and sr are the same as in feature.py
sr = 44100
nfft = 2048
frames_1_sec = int(sr/(nfft/2.0))

print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

# Folder for saving model and training curves
__models_dir = 'models/'
utils.create_folder(__models_dir)

# CRNN model definition
cnn_nb_filt = 128           # CNN filter size
cnn_pool_size = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
rnn_nb = [32, 32]           # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
fc_nb = [32]                # Number of FC nodes.  Length of fc_nb =  number of FC layers
dropout_rate = 0.5          # Dropout after each layer
print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
    cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

avg_er = list()
avg_f1 = list()

## -1 0 2 4 10 denote the original, brakes, children, people speaking and allthreeclass
for fold in [0]:
    directory = '/data/users/21799506/Data/PRL2018/Evaluation/models_Temp'
    X, Y, X_test, Y_test = load_data(feat_folder, is_mono)
    X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)
    

    ###################### Construct the model below
    ###################### Construct the model above
    # Load model
    model = get_model(X, Y, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb)
    #exit(0)
    # Training
    best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
    tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch
    f1_overall_1sec_list_eval,er_overall_1sec_list_eval = [0] * nb_epoch, [0] * nb_epoch
    posterior_thresh = 0.5
    for i in range(nb_epoch):
        print('Epoch : {} '.format(i), end='')
        hist = model.fit(
            X, Y,
            batch_size=batch_size,
            validation_split=0.2,
            epochs=1,
            verbose=2
        )
        X_eval = X[int(0.8*X.shape[0]):X.shape[0],:,:,:]
        Y_eval = Y[int(0.8*X.shape[0]):X.shape[0],:,:]
        val_loss[i] = hist.history.get('val_loss')[-1]
        tr_loss[i] = hist.history.get('loss')[-1]
        

        ## performance on the development dataset
        pred_eval = model.predict(X_eval)
        pred_thresh_eval = pred_eval > posterior_thresh
        score_list_eval = metrics.compute_scores(pred_thresh_eval, Y_eval, frames_in_1_sec=frames_1_sec)
        f1_overall_1sec_list_eval[i] = score_list_eval['f1_overall_1sec']
        er_overall_1sec_list_eval[i] = score_list_eval['er_overall_1sec'] 
        test_pred_cnt_eval = np.sum(pred_thresh_eval, 2)
        Y_test_cnt_eval = np.sum(Y_eval, 2)
        conf_mat_eval = confusion_matrix(Y_test_cnt_eval.reshape(-1), test_pred_cnt_eval.reshape(-1))
        conf_mat_eval = conf_mat_eval / (utils.eps + np.sum(conf_mat_eval, 1)[:, None].astype('float'))
        ## performance on the evaluation  dataset        
        pred = model.predict(X_test)
        pred_thresh = pred > posterior_thresh
        score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)

        f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
        er_overall_1sec_list[i] = score_list['er_overall_1sec']
        test_pred_cnt = np.sum(pred_thresh, 2)
        Y_test_cnt = np.sum(Y_test, 2)


        test_pred_ind, Y_test_ind = GetEvents(pred,Y_test,posterior_thresh)
        conf_mat = confusion_matrix(Y_test_ind, test_pred_ind, labels = [0, 1,2,3,4,5,6])        
        print('Epoch_{}:F1_{}   ER_{}'.format(i,score_list['f1_overall_1sec'],score_list['er_overall_1sec']))   
        print(conf_mat)
        conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))
        print(conf_mat)
        model_path =  '/data/users/21799506/Data/PRL2018/Evaluation/models_Temp/model_' + str(i) + '.h5'
        model.save(model_path)        
        plot_functions(nb_epoch, tr_loss, val_loss, f1_overall_1sec_list_eval,er_overall_1sec_list_eval, f1_overall_1sec_list, er_overall_1sec_list)    

        attention_vector = get_activations(model,X_test,print_shape_only=True,layer_name='attention_vec')[0]
        conv_vector = get_activations(model,X_test,print_shape_only=True,layer_name='conv_vec')[0]
        savevector = attention_vector.reshape(-1,256,256)
        name = '/data/users/21799506/Data/PRL2018/Evaluation/AttentionVector_Temp/attention_vector' + str(i)
        sio.savemat(name,{'array_atten':savevector})    
        name = '/data/users/21799506/Data/PRL2018/Evaluation/AttentionVector_Temp/conv_vector' + str(i)
        sio.savemat(name,{'array_conv':conv_vector})          
        
    lossname = '/data/users/21799506/Data/PRL2018/Evaluation/models_Temp/performance'
    sio.savemat(lossname,{'arr_0':tr_loss,'arr_1':val_loss,'arr_2':f1_overall_1sec_list,'arr_3':er_overall_1sec_list})
