# Functions used during DDF generation

import os
import torch
from baseline_model.network import build_model
from utils.transform import Transform2Transform,TransformAccumulation
from utils.plot_functions import reference_image_points,read_calib_matrices


class Prediction():  

    def __init__(self, parameters,model_name,data_path_calib,model_path):
        self.parameters = parameters
        os.environ["CUDA_VISIBLE_DEVICES"] = self.parameters['gpu_ids']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_pairs=torch.tensor([[0,1]]).to(self.device)
        self.model_name = model_name
        self.model_path = model_path
        self.tform_calib_scale,self.tform_calib_R_T, self.tform_calib = read_calib_matrices(data_path_calib)
        # image points coordinates in image coordinate system, all pixel points
        self.image_points = reference_image_points([480, 640],[480, 640]).to(self.device)
        
        # transform prediction into 4*4 transformation matrix
        self.transform_to_transform = Transform2Transform(
            pred_type=self.parameters['PRED_TYPE'],
            num_pairs=self.data_pairs.shape[0],
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(self.device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(self.device),
            tform_image_pixel_to_mm = self.tform_calib_scale.to(self.device)
            )
        # accumulate transformation
        self.transform_accumulation = TransformAccumulation(
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(self.device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(self.device),
            tform_image_pixel_to_image_mm = self.tform_calib_scale.to(self.device)
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


    def generate_pred_values(self, frames,landmark):
        # calculate DDFs
        frames = torch.tensor(frames)[None,...].to(self.device)
        frames = frames/255
        landmark = torch.from_numpy(landmark)
        # Global displacement vectors for pixel reconstruction
        transformation_local, transformation_global, pred_global_allpts_DDF = self.cal_pred_global_allpts(frames)
        # Global displacement vectors for landmark reconstruction
        pred_global_landmark_DDF = self.cal_pred_global_landmark(transformation_global,landmark)
        # Local displacement vectors for pixel reconstruction
        pred_local_allpts_DDF = self.cal_pred_local_allpts(transformation_local)
        # Local displacement vectors for landmark reconstruction
        pred_local_landmark_DDF = self.cal_pred_local_landmark(transformation_local,landmark)
        
        return pred_global_allpts_DDF, pred_global_landmark_DDF, pred_local_allpts_DDF, pred_local_landmark_DDF

    def cal_pred_global_allpts(self,frames):
        # generate global displacement vectors for pixel reconstruction
        # NOTE: when NUM_SAMPLES is larger than 2, the coords of the last (NUM_SAMPLES-2) frames will not be generated
        predictions_global_allpts = torch.zeros((frames.shape[1]-1,3,self.image_points.shape[-1]))
        # local transformation, i.e., transformation from current frame to the immediate previous frame
        transformation_local = torch.zeros(frames.shape[1]-1,4,4)
        # global transformation, i.e., transformation from current frame to the first frame
        transformation_global = torch.zeros(frames.shape[1]-1,4,4)

        prev_transf = torch.eye(4).to(self.device)
        idx_f0 = 0  # this is the reference frame for network prediction
        Pair_index = 0 # select which pair of prediction to use
        interval_pred = torch.squeeze(self.data_pairs[Pair_index])[1] - torch.squeeze(self.data_pairs[Pair_index])[0]

        while True:
            
            frames_sub = frames[:,idx_f0:idx_f0 + self.parameters['NUM_SAMPLES'], ...].to(self.device)
            with torch.no_grad():
                outputs = self.model(frames_sub)
                # transform prediction into 4*4 transformtion matrix, to be accumulated
                preds_transf = self.transform_to_transform(outputs)[0,Pair_index,...] # use the transformtion from image 1 to 0
                transformation_local[idx_f0] = preds_transf
                # calculate global transformation and global reconstruction coordinates
                pts_cord, prev_transf = self.transform_accumulation(prev_transf,preds_transf)
                transformation_global[idx_f0] = prev_transf
                predictions_global_allpts[idx_f0] = pts_cord[0:3,...].cpu()


            idx_f0 += interval_pred
            # NOTE: Due to this break, when NUM_SAMPLES is larger than 2, the coords of the last (NUM_SAMPLES-2) frames will not be generated
            if (idx_f0 + self.parameters['NUM_SAMPLES']) > frames.shape[1]:
                break

        if self.parameters['NUM_SAMPLES'] > 2:
            # NOTE: As not all frames will be reconstructed, we could use interpolation or some other methods to fill the missing frames
            # This baseline code provides a simple way to fill the missing frames, by using the last reconstructed frame to fill the missing frames
            transformation_local[idx_f0:,...] = torch.eye(4).expand(transformation_local[idx_f0:,...].shape[0],-1,-1)
            transformation_global[idx_f0:,...] = transformation_global[idx_f0-1].expand(transformation_global[idx_f0:,...].shape[0],-1,-1)
            predictions_global_allpts[idx_f0:,...] = predictions_global_allpts[idx_f0-1].expand(predictions_global_allpts[idx_f0:,...].shape[0],-1,-1)

        # calculate DDF in mm, which is the displacement from current frame to the first frame
        predictions_global_allpts_DDF = predictions_global_allpts.numpy() -torch.matmul(self.tform_calib_scale.to(self.device),self.image_points)[0:3,:].expand(predictions_global_allpts.shape[0],-1,-1).cpu().numpy()
        
        return transformation_local, transformation_global,predictions_global_allpts_DDF
        
    def cal_pred_global_landmark(self,transformation_global,landmark):
        # generate global displacement vectors for landmark reconstruction
        # transformation_global: transformation from current frame to the first frame
        # landmark: coordinates of landmark points in image coordinate system (in pixel)

        pred_global_landmark = torch.zeros(3,len(landmark))
        for i in range(len(landmark)):  
            # point coordinate in image coordinate system (in pixel)  
            pts_coord = torch.cat((landmark[i][1:], torch.FloatTensor([0,1])),axis = 0).to(self.device)
            # calculate DDF in mm, displacement from current frame to the first frame
            pred_global_landmark[:,i] = torch.matmul(transformation_global[landmark[i][0]-1].to(self.device),torch.matmul(self.tform_calib_scale.to(self.device),pts_coord))[0:3]-torch.matmul(self.tform_calib_scale.to(self.device),pts_coord)[0:3]
    
        pred_global_landmark_DDF = pred_global_landmark.cpu().numpy()

        return pred_global_landmark_DDF
    
    def cal_pred_local_allpts(self,transformation_local):
        # generate local displacement vectors for pixel reconstruction

        # coordinates of points in current frame, with respect to the immediately previous frame 
        prediction_local_allpts = torch.matmul(transformation_local.to(self.device),torch.matmul(self.tform_calib_scale.to(self.device),self.image_points))
        # calculate DDF in mm, displacement from current frame to the immediately previous frame
        prediction_local_allpts_DDF = prediction_local_allpts[:,0:3,:]-torch.matmul(self.tform_calib_scale.to(self.device),self.image_points)[0:3,:].expand(prediction_local_allpts.shape[0],-1,-1)
        prediction_local_allpts_DDF = prediction_local_allpts_DDF.cpu().numpy()
        
        return prediction_local_allpts_DDF

    def cal_pred_local_landmark(self,transformation_local,landmark):
        # generate local displacement vectors for landmark reconstruction

        pred_local_landmark = torch.zeros(3,len(landmark))
        for i in range(len(landmark)):  
            # point coordinate in image coordinate system (in pixel)  
            pts_coord = torch.cat((landmark[i][1:], torch.FloatTensor([0,1])),axis = 0).to(self.device)
            # calculate DDF in mm, displacement from current frame to the immediately previous frame
            pred_local_landmark[:,i] = torch.matmul(transformation_local[landmark[i][0]-1].to(self.device),torch.matmul(self.tform_calib_scale.to(self.device),pts_coord))[0:3]-torch.matmul(self.tform_calib_scale.to(self.device),pts_coord)[0:3]
    
        pred_local_landmark_DDF = pred_local_landmark.cpu().numpy()

        return pred_local_landmark_DDF
    
    def type_dim(self,label_pred_type, num_points=None, num_pairs=1):
        # return the dimension of the label or prediction, based on the type of label or prediction
        type_dim_dict = {
            "transform": 12,
            "parameter": 6,
            "point": num_points*3,
            "quaternion": 7
        }
        return type_dim_dict[label_pred_type] * num_pairs  # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]

   