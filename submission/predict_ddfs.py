# this is the function for generating DDF
import os
import json
from baseline_model.Prediction import Prediction

def predict_ddfs(frames,landmark,data_path_calib):
    """
    NOTE:
    Here is just an example.
    Fill in below with your own function.
    Requirement details can be found at : https://github-pages.ucl.ac.uk/tus-rec-challenge/submission.html
    
    Args:
        frames (numpy.ndarray): shape=(N, 480, 640),frames in the scan, where N is the number of frames in this scan
        landmark (numpy.ndarray): shape=(20,3), denoting the location of landmark. For example, a landmark with location of (10,200,100) denotes the landmark on the 10th frame, with coordinate of (200,100)
        data_path_calib (str): path to calibration matrix

    Returns:
        pred_global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame)
        pred_global_landmark_DDF (numpy.ndarray): shape=(3, 20), global DDF for landmark  
        pred_local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame) 
        pred_local_landmark_DDF (numpy.ndarray): shape=(3, 20), local DDF for landmark
    
    """
    
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
    pred_local_landmark_DDF = prediction.generate_prediction_DDF(frames,landmark)

    return pred_global_allpts_DDF,pred_global_landmark_DDF,pred_local_allpts_DDF,pred_local_landmark_DDF

