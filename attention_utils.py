import keras.backend as K
import numpy as np
import scipy.io as sio

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, attention_column))
    x[:, 0:attention_column] = y[:, 0:attention_column]
    return x, y

def get_real_data_class():
    DatDim = 40
    AggreNum = 10
    Data = sio.loadmat('TrainData_fold1_mel_scale.mat')
    Data = Data['array']
    TrainData = np.zeros((Data.shape[0],DatDim*AggreNum))
    TrainData = Data[:,0:DatDim*AggreNum]
    TrainLabel = np.zeros((Data.shape[0],6))
    TrainLabel = Data[:,DatDim*AggreNum:DatDim*AggreNum+6]

    Data = sio.loadmat('TestData_fold1_mel_scale.mat')
    Data = Data['array']
    TestData = np.zeros((Data.shape[0],DatDim*AggreNum))
    TestData = Data[:,0:DatDim*AggreNum]
    TestLabel = np.zeros((Data.shape[0],6))
    TestLabel = Data[:,DatDim*AggreNum:DatDim*AggreNum+6]

    TrainData = TrainData[:,0:40*AggreNum]
    TestData = TestData[:,0:40*AggreNum]

    return TrainData,TrainLabel,TestData,TestLabel

def get_real_data_reg():
    DatDim = 40
    AggreNum = 10
    Data = sio.loadmat('TrainData_fold1_mel_scale.mat')
    Data = Data['array']
    TrainData = np.zeros((Data.shape[0],DatDim*AggreNum))
    TrainData = Data[:,0:DatDim*AggreNum]
    TrainLabel = np.zeros((Data.shape[0],6))
    TrainLabel = Data[:,DatDim*10+6:DatDim*10+12]

    Data = sio.loadmat('TestData_fold1_mel_scale.mat')
    Data = Data['array']
    TestData = np.zeros((Data.shape[0],DatDim*AggreNum))
    TestData = Data[:,0:DatDim*AggreNum]
    TestLabel = np.zeros((Data.shape[0],6))
    TestLabel = Data[:,DatDim*10+6:DatDim*10+12]

    TrainData = TrainData[:,0:40*2]
    TestData = TestData[:,0:40*2]

    return TrainData,TrainLabel,TestData,TestLabel    

def get_real_virtual_data():
    DatDim = 40
    AggreNum = 10
    Use = 2
    import h5py
    Data = h5py.File('H:/My Documents/Tools/RandomForestRegression/CVAE/gan_ac/Extended1356_90000.mat')
    Data = Data['array']
    #Data = sio.loadmat('gan_virtualData/Extended1_90000.mat')
    #Data = Data['array']
    #Data = Data.T
    TrainData = np.zeros((Data.shape[0],DatDim*AggreNum))
    TrainData = Data[:,0:DatDim*AggreNum]
    TrainLabel = np.zeros((Data.shape[0],6))
    TrainLabel = Data[:,DatDim*AggreNum:DatDim*AggreNum+6]

    Data = sio.loadmat('TestData_fold1_mel_scale.mat')
    Data = Data['array']
    TestData = np.zeros((Data.shape[0],DatDim*AggreNum))
    TestData = Data[:,0:DatDim*AggreNum]
    TestLabel = np.zeros((Data.shape[0],6))
    TestLabel = Data[:,DatDim*AggreNum:DatDim*AggreNum+6]

    TrainData = TrainData[:,0:40*Use]
    TestData = TestData[:,0:40*Use]

    return TrainData,TrainLabel,TestData,TestLabel    

def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y
