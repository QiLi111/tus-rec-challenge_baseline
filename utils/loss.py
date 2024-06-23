# Loss function and metrics

from utils.transform import TransformAccumulation
from utils.transform import LabelTransform
import numpy as np
from itertools import combinations

class PointDistance:
    # calculate the distance between two sets of points
    def __init__(self,paired=True,):
        
        self.paired = paired
    
    def __call__(self,preds,labels):
        if self.paired:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean(dim=(0,2))
        else:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean()
        
