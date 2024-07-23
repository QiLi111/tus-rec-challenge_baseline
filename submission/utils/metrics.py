# calculate metrics
import numpy as np

def cal_dist(label,pred,mode = 'all'):
    # calculate distance between ground truth and prediction coordinates

    if mode == 'all':
        dist = np.sqrt(((label-pred)**2).sum(axis=1)).mean()
    elif mode == 'landmark':
        dist = np.sqrt(((label-pred)**2).sum(axis=0)).mean()

    return dist
