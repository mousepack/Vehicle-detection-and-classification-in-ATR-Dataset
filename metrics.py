import numpy as np
from keras import backend as K
def point_dice_coef(y_true, y_pred):
    smooth = .0001

    y_true_f = y_true.flatten()
    y_true_f_neg = -1*((y_true-1.0).flatten())

    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    intersection_neg = np.sum(y_true_f_neg * y_pred_f)

    return ((intersection_neg + smooth) / (intersection + smooth))

def IoU(y_true,y_pred):
    smooth = .0001
    y_true_f = (y_true.flatten()>.5).astype(np.uint8)
    y_pred_f = (y_pred.flatten()>.5).astype(np.uint8)
    overlap = y_true_f*y_pred_f
    union = ((y_true_f+y_pred_f)>0).astype(np.uint8)
    return(np.sum(overlap)/(np.sum(union)+smooth))

def dice_coef(y_true, y_pred):
    smooth = .00001
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def recall(y_true, y_pred):
    smooth = 1
    # import ipdb;ipdb.set_trace()
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + smooth)

    return recall


def precision(y_true, y_pred):
    smooth = 1
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + smooth)

    return precision

# def IoU(y_true, y_pred):
#     smooth = 1
#     y_true_f = y_true.flatten() 
#     y_pred_f = y_pred.flatten()

#     pre = precision(y_true_f, y_pred_f)
#     rec = recall(y_true_f, y_pred_f)

#     return 2 * ((pre * rec) / (pre + rec + smooth))


def f1(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten() 
    y_pred_f = y_pred.flatten()

    pre = precision(y_true_f, y_pred_f)
    rec = recall(y_true_f, y_pred_f)

    return 2 * ((pre * rec) / (pre + rec + smooth))
