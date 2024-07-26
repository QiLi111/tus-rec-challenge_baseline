# Functions used during DDF generation

import os
import torch
from baseline_model.network import build_model
from utils.transform import Transforms,TransformAccumulation
from utils.plot_functions import reference_image_points,read_calib_matrices
from utils.Transf2DDFs import cal_global_allpts,cal_global_landmark,cal_local_allpts,cal_local_landmark


class Prediction():  

    def __init__(self, parameters,model_name,data_path_calib,model_path):
        self.parameters = parameters
        os.environ["CUDA_VISIBLE_DEVICES"] = self.parameters['gpu_ids']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_pairs=torch.tensor([[0,1]]).to(self.device)
        self.model_name = model_name
        self.model_path = model_path
        self.tform_calib_scale,self.tform_calib_R_T, self.tform_calib = read_calib_matrices(data_path_calib)
        self.tform_calib_scale,self.tform_calib_R_T, self.tform_calib = self.tform_calib_scale.to(self.device),self.tform_calib_R_T.to(self.device), self.tform_calib.to(self.device)
        # image points coordinates in image coordinate system, all pixel points
        self.image_points = reference_image_points([480, 640],[480, 640]).to(self.device)
        
        # transform prediction into 4*4 transformation matrix
        self.transforms = Transforms(
            pred_type=self.parameters['PRED_TYPE'],
            num_pairs=self.data_pairs.shape[0],
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            tform_image_pixel_to_mm = self.tform_calib_scale
            )
        # accumulate transformation
        self.transform_accumulation = TransformAccumulation(
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            tform_image_pixel_to_image_mm = self.tform_calib_scale
            )

        self.pred_dim = self.type_dim(self.parameters['PRED_TYPE'], self.image_points.shape[1], self.data_pairs.shape[0])
        self.label_dim = self.type_dim(self.parameters['LABEL_TYPE'], self.image_points.shape[1], self.data_pairs.shape[0])
        
        self.model = build_model(
            self.parameters,
            in_frames = self.parameters['NUM_SAMPLES'],
            pred_dim = self.pred_dim,
            ).to(self.device)
        ## load the model
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, self.model_name),map_location=torch.device(self.device)))
        self.model.train(False)


    def generate_prediction_DDF(self, frames,landmark):
        # calculate DDFs
        frames = torch.tensor(frames)[None,...].to(self.device)
        frames = frames/255
        landmark = torch.from_numpy(landmark)

        # predict global and local transformations, from model
        transformation_global, transformation_local = self.cal_pred_transformations(frames)
        transformation_global, transformation_local = transformation_global.to(self.device), transformation_local.to(self.device)
        # Global displacement vectors for pixel reconstruction
        pred_global_allpts_DDF = cal_global_allpts(transformation_global,self.tform_calib_scale,self.image_points)
        # Global displacement vectors for landmark reconstruction
        pred_global_landmark_DDF = cal_global_landmark(transformation_global,landmark,self.tform_calib_scale)
        # Local displacement vectors for pixel reconstruction
        pred_local_allpts_DDF = cal_local_allpts(transformation_local,self.tform_calib_scale,self.image_points)
        # Local displacement vectors for landmark reconstruction
        pred_local_landmark_DDF = cal_local_landmark(transformation_local,landmark,self.tform_calib_scale)
        
        return pred_global_allpts_DDF, pred_global_landmark_DDF, pred_local_allpts_DDF, pred_local_landmark_DDF

    def cal_pred_transformations(self,frames):
        """
        predict global and local transformations from model

        Args:
            frames (torch.Tensor): shape=(1, N, H, W), all frames in the scan, where N denotes the number of frames in the scan, H and W denote the height and width of a frame. 

        Returns:
            transformation_global (torch.Tensor): shape=(N-1, 4, 4), global transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the first frame
            transformation_local (torch.Tensor): shape=(N-1, 4, 4), local transformations for each frame in the scan; each transformation denotes the transformation from the current frame to the previous frame

        """

        # NOTE: when NUM_SAMPLES is larger than 2, the transformations of the last (NUM_SAMPLES-2) frames will not be generated
        # local transformation, i.e., transformation from current frame to the immediate previous frame
        transformation_local = torch.zeros(frames.shape[1]-1,4,4)
        # global transformation, i.e., transformation from current frame to the first frame
        transformation_global = torch.zeros(frames.shape[1]-1,4,4)

        prev_transf = torch.eye(4).to(self.device)
        idx_f0 = 0  # this is the reference frame for network prediction
        Pair_index = 0 # select which pair of prediction to use
        interval_pred = torch.squeeze(self.data_pairs[Pair_index])[1] - torch.squeeze(self.data_pairs[Pair_index])[0]

        while True:  
            frames_sub = frames[:,idx_f0:idx_f0 + self.parameters['NUM_SAMPLES'], ...]
            with torch.no_grad():
                outputs = self.model(frames_sub)
                # transform prediction into 4*4 transformtion matrix, to be accumulated
                preds_transf = self.transforms(outputs)[0,Pair_index,...] 
                transformation_local[idx_f0] = preds_transf
                # calculate global transformation
                prev_transf = self.transform_accumulation(prev_transf,preds_transf)
                transformation_global[idx_f0] = prev_transf

            idx_f0 += interval_pred
            # NOTE: Due to this break, when NUM_SAMPLES is larger than 2, the transformations of the last (NUM_SAMPLES-2) frames will not be generated
            if (idx_f0 + self.parameters['NUM_SAMPLES']) > frames.shape[1]:
                break

        if self.parameters['NUM_SAMPLES'] > 2:
            # NOTE: As not all frames will be reconstructed, we could use interpolation or some other methods to fill the missing frames
            # This baseline code provides a simple way to fill the missing frames, by using the last reconstructed frame to fill the missing frames
            transformation_local[idx_f0:,...] = torch.eye(4).expand(transformation_local[idx_f0:,...].shape[0],-1,-1)
            transformation_global[idx_f0:,...] = transformation_global[idx_f0-1].expand(transformation_global[idx_f0:,...].shape[0],-1,-1)
        
        return transformation_global,transformation_local
   
    def type_dim(self,label_pred_type, num_points=None, num_pairs=1):
        # return the dimension of the label or prediction, based on the type of label or prediction
        type_dim_dict = {
            "transform": 12,
            "parameter": 6,
            "point": num_points*3,
            "quaternion": 7
        }
        return type_dim_dict[label_pred_type] * num_pairs  # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]
