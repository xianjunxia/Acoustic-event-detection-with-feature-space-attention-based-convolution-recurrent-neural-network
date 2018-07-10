import numpy as np
import utils
#####################
# Scoring functions
#
# Code blocks taken from Toni Heittola's repository: http://tut-arg.github.io/sed_eval/
#
# Implementation of the Metrics in the following paper:
# Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 'Metrics for polyphonic sound event detection',
# Applied Sciences, 6(6):162, 2016
#####################


def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + utils.eps)
    recall = float(TP) / float(Nref + utils.eps)
    f1_score = 2 * prec * recall / (prec + recall + utils.eps)
    return f1_score


def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()
    ER = (max(Nref, Nsys) - TP) / (Nref + 0.0)
    return ER


def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    new_size = int(O.shape[0] / block_size)
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_scores(pred, y, frames_in_1_sec=50):
    scores = dict()
    scores['f1_overall_1sec'] = f1_overall_1sec(pred, y, frames_in_1_sec)
    scores['er_overall_1sec'] = er_overall_1sec(pred, y, frames_in_1_sec)
    return scores

def compute_scores_classwise(pred, y, seq,frames_in_1_sec=50):
    scores = dict()
    for c in range(6):
        pred2, y2 = GetSingClassPrediction(pred=pred,y=y,c=c)
        name1 = 'f1_overall_1sec_' + str(c)
        name2 = 'er_overall_1sec_' + str(c)
        scores[name1] = f1_overall_1sec(pred2, y2, frames_in_1_sec)
        scores[name2] = er_overall_1sec(pred2, y2, frames_in_1_sec)
    return scores 

def GetSingClassPrediction(pred,y,c):
    pred2 = pred.reshape(pred.shape[0]*pred.shape[1],pred.shape[2])
    y2 = y.reshape(y.shape[0]*y.shape[1],y.shape[2])
    cnt = 0
    for s  in range (pred2.shape[0]):            
        if(pred2[s,c] == True):
            cnt = cnt + 1
            pred2[s,:] = False
            pred2[s,c] = True
            y2[s,:] = False
            y2[s,c] = True
    print(cnt)
    pred2 = pred2.reshape(pred.shape[0],pred.shape[1],pred.shape[2]) 
    y2 = y2.reshape(pred.shape[0],pred.shape[1],pred.shape[2])
    return pred2, y2


