# this is the function for generating DDF
import os
import torch
import json
from baseline_model.Prediction import Prediction

def predict_ddfs(frames,landmark,data_path_calib):
    '''
    Note:
    Here is just an example.
    Fill in below with your own function.
    Required Input:
                    frames: frames in the scan, numpy array with the shape of [N,480,640], where N is the number of frames in this scan
                    landmark: numpy array in a shape of [20,3], denoting the location of landmark. For example, a landmark with location of [10,200,100] denotes the landmark on the 10th frame, with coordinate of [200,100]
                    data_path_calib: path to calibration matrix.
    Required Output: 
                    four kinds of DDFs: GP,GL,LP,LL
    '''
    # path to the baseline model
    model_path = os.path.dirname(os.path.realpath(__file__))+'/'+'baseline_model'
    # parameters used in baseline code
    with open(model_path+'/'+'config.json') as f:
        parameters = json.load(f)

    prediction = Prediction(parameters,'model_weights',data_path_calib,model_path)
    # generate 4 DDFs for the scan
    pred_global_allpts_DDF,\
    pred_global_landmark_DDF,\
    pred_local_allpts_DDF,\
    pred_local_landmark_DDF = prediction.generate_pred_values(frames,landmark)

    return pred_global_allpts_DDF,pred_global_landmark_DDF,pred_local_allpts_DDF,pred_local_landmark_DDF

