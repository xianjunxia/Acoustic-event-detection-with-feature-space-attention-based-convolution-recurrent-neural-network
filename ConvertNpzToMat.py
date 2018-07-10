import numpy as np
import scipy.io as sio
feat_file_fold = '/data/users/21799506/Data/PRL2018/Evaluation/feat/' + 'mbe_mon_fold0' + '.npz'
dmp = np.load(feat_file_fold)
_X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
name = '/data/users/21799506/Data/PRL2018/Evaluation/Test.mat'
sio.savemat(name,{'testX':_X_test,'testY':_Y_test})
