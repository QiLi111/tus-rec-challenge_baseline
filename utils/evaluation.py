import os
from matplotlib import pyplot as plt
import torch,h5py
import numpy as np
import torch.nn.functional as F
from utils.network import build_model
from utils.data_process_functions import *
from utils.transform import LabelTransform, PredictionTransform, PointTransform
from utils.loss import PointDistance
from utils.funs import *



class Evaluation():  

    def __init__(self, opt,device, dset,model_name,data_pairs,saved_folder):
        self.opt = opt
        self.device = device
        self.dset = dset
        self.saved_folder = saved_folder
        self.model_name = model_name
        self.data_pairs = data_pairs.to(device)
        self.tform_calib_scale,self.tform_calib_R_T, self.tform_calib = read_calib_matrices(opt.FILENAME_CALIB)
        # image points coordinates on image coordinate system
        self.image_points = reference_image_points(self.dset[0][0].shape[1:],2).to(device)

        self.transform_label = LabelTransform(
            self.opt.LABEL_TYPE,
            pairs=self.data_pairs,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(device)
            )

        self.transform_prediction = PredictionTransform(
            opt.PRED_TYPE,
            opt.LABEL_TYPE,
            num_pairs=self.data_pairs.shape[0],
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(device)
            )

        self.pred_dim = type_dim(self.opt.PRED_TYPE, self.image_points.shape[1], self.data_pairs.shape[0])
        self.label_dim = type_dim(self.opt.LABEL_TYPE, self.image_points.shape[1], self.data_pairs.shape[0])
        
        
        ## load the model
        self.model = build_model(
            self.opt,
            in_frames = self.opt.NUM_SAMPLES,
            pred_dim = self.pred_dim,
            ).to(device)
        
        self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', self.model_name),map_location=torch.device(self.device)))
        self.model.train(False)

        # evaluation metrics initialisation

        self.Global_AllPts = [] # Global displacement vectors for pixel reconstruction
        self.Global_Landmark = [] # Global displacement vectors for landmark reconstruction
        self.Local_AllPts = [] # Local displacement vectors for pixel reconstruction
        self.Local_Landmark = [] # Local displacement vectors for landmark reconstruction
        # store prediction for all scans in .h5 file    
        self.Global_AllPts = h5py.File(os.path.join(self.saved_folder,"Global_AllPts.h5"),'a')
        self.Global_Landmark = h5py.File(os.path.join(self.saved_folder,"Global_Landmark.h5"),'a')
        self.Local_AllPts = h5py.File(os.path.join(self.saved_folder,"Local_AllPts.h5"),'a')
        self.Local_Landmark = h5py.File(os.path.join(self.saved_folder,"Local_Landmark.h5"),'a')
        # store ground truth for all scans in .h5 file
        # when testing, this is not visible for participants
        self.labels = h5py.File(os.path.join(self.saved_folder,"labels.h5"),'a')
    
    def generate_keys(self,scan_index):
        # generate keys for the prediction h5 file, each key representing a scan in one subject, initialised with []
        # this is to indicate how the dataset looks like in test sets
        initial_value = []
        _,_,indices, scan_name = self.dset[scan_index]
        self.Global_AllPts.create_dataset('/sub%03d_%s' % (indices[0],scan_name), len(initial_value), dtype=initial_value.dtype, data=initial_value)
        self.Global_Landmark.create_dataset('/sub%03d_%s' % (indices[0],scan_name), len(initial_value), dtype=initial_value.dtype, data=initial_value)
        self.Local_AllPts.create_dataset('/sub%03d_%s' % (indices[0],scan_name), len(initial_value), dtype=initial_value.dtype, data=initial_value)
        self.Local_Landmark.create_dataset('/sub%03d_%s' % (indices[0],scan_name), len(initial_value), dtype=initial_value.dtype, data=initial_value)
    
    def close_files(self):
        self.Global_AllPts.flush()
        self.Global_AllPts.close()
        self.Global_Landmark.flush()
        self.Global_Landmark.close()
        self.Local_AllPts.flush()
        self.Local_AllPts.close()
        self.Local_Landmark.flush()
        self.Local_Landmark.close()
        self.labels.flush()
        self.labels.close()

    def calculate_GT_DDF(self, scan_index):
        # calculate DDF of ground truth - label
        # when testing, this is not visible for participants
        frames, tforms, indices, scan_name = self.dset[scan_index]
        frames, tforms = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms])
        tforms_inv = torch.linalg.inv(tforms)

        data_pairs_all = data_pairs_adjacent(frames.shape[1])
        data_pairs_all=torch.tensor(data_pairs_all) 
        transform_label_all = LabelTransform(
            "point",
            pairs=self.data_pairs,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(self.device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(self.device)
            )
        
        tforms_each_frame2frame0_gt = transform_label_all(tforms, tforms_inv)
        labels = torch.matmul(tforms_each_frame2frame0_gt,self.image_points)[:,:,0:3,...]
        
        self.labels.create_dataset('/sub%03d_%s' % (indices[0],scan_name), labels.shape, dtype=labels.dtype, data=labels)


    def calculate_predicted_DDF(self, scan_index,saved_folder):
        # calculate DDF of prediction, and then save it into the key-values= pair in prediction h5 file
        # when testing, only images and the keys (stored in .h5 file) are avaliable, tforms are not avaliable
        # one way to get the structure of the test set is to use the keys in the prediction h5 file
        dataset_test = h5py.File(os.path.join(self.opt.PREDICTION_PATH,'Global_AllPts'), 'r')
        scan_name_all = dataset_test.keys() # data structure of the test set
        for i in range(scan_name_all):
            scan = h5py.File(os.path.join(self.opt.DATA_PATH,scan_name_all[i].split('_')[0][3:],scan_name_all[i].split('_')[1]), 'r')

            frames, tforms, scan_name = self.dset[scan_index]
            frames, tforms = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms])
            tforms_inv = torch.linalg.inv(tforms)
            frames = frames/255
            saved_folder = saved_folder

            idx = 0
            while True:
                idx_f0 = 0  #  this is the reference frame for network prediction
                idx_p0 = idx_f0   # this is the reference frame for transformaing others to
                idx_p1 = idx_f0 + torch.squeeze(self.data_pairs[0])[1] # an example to use transformation from frame 1 to frame 0
                interval_pred = torch.squeeze(self.data_pairs[0])[1] - torch.squeeze(self.data_pairs[0])[0]

                if (idx + self.opt.NUM_SAMPLES) > frames.shape[1]:
                    break

                frames_sub = frames[:,idx:idx + self.opt.NUM_SAMPLES, ...]
                tforms_sub = tforms[:,idx:idx + self.opt.NUM_SAMPLES, ...]
                tforms_inv_sub = tforms_inv[:,idx:idx + self.opt.NUM_SAMPLES, ...]

                with torch.no_grad():
                    outputs = self.model(frames_sub)
                    pred_transfs = self.transform_prediction(outputs)

                    # make the predicted transformations are based on frame 0
                    predframe0 = torch.eye(4,4)[None,...].repeat(pred_transfs.shape[0],1, 1,1).to(self.device)
                    pred_transfs = torch.cat((predframe0,pred_transfs),1)

                    # calculate local transformation
                    pred_transfs_0 = pred_transfs[:,0:-1,...]
                    pred_transfs_1 = pred_transfs[:,1:,...]
                    pred_transfs_local = torch.matmul(torch.linalg.inv(pred_transfs_0),pred_transfs_1)
                    pred_transfs_local = torch.matmul(transf_0,pred_transfs_local)
                    pred_transfs_local = torch.cat((predframe0,pred_transfs_local),1)
                


                if idx !=0:
                    # if not the first sub-sequence, should be transformed into frame 0
                    tforms_each_frame2frame0_gt_sub = torch.matmul(tform_last_frame[None,...],tforms_each_frame2frame0_gt_sub)
                    pred_transfs = torch.matmul(tform_last_frame_pred[None,...],pred_transfs) 

                    tforms_each_frame2frame0_gt_sub_local = torch.matmul(tform_last_frame_local[None,...],tforms_each_frame2frame0_gt_sub_local)
                    pred_transfs_local = torch.matmul(tform_last_frame_pred_local[None,...],pred_transfs_local) 

                
                tform_last_frame = tforms_each_frame2frame0_gt_sub[:,-1,...]
                tform_last_frame_pred = pred_transfs[:,-1,...]

                tform_last_frame_local = tforms_each_frame2frame0_gt_sub_local[:,-1,...]
                tform_last_frame_pred_local = pred_transfs_local[:,-1,...]

                # obtain the coordinates of each frame, using frame 0 as the reference frame
                if self.opt.img_pro_coord == 'img_coord':
                    labels_gt_sub = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]
                    # transformtion to points
                    pred_pts_sub = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]

                    labels_gt_sub_local = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0_gt_sub_local,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]
                    pred_pts_sub_local = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(pred_transfs_local,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]


                    
                
                else:
                    labels_gt_sub = torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
                    # transformtion to points
                    pred_pts_sub = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
                    
                    labels_gt_sub_local = torch.matmul(tforms_each_frame2frame0_gt_sub_local,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
                    # transformtion to points
                    pred_pts_sub_local = torch.matmul(pred_transfs_local,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]



                    # for coordinates system change use
                    ori_pts_sub = torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    pre_sub = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    # obtain points in optimised coodinates system, and the coorsponding transformation matrix
                    labels_gt_sub_opt, pred_pts_sub_opt,convR_batched,minxyz_all = self.ConvPose(labels_gt_sub, ori_pts_sub, pre_sub, 'auto_PCA',self.device)

                    ori_pts_sub_local = torch.matmul(tforms_each_frame2frame0_gt_sub_local,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    pre_sub_local = torch.matmul(pred_transfs_local,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    


                # this is the warpped prediction, represented in optimised coordinates system
                # as there will be common frame in two US sequence, when produce DDF, only one DDF should be generated
                if idx == 0:
                    labels_gt_sub_non_overlap_opt = labels_gt_sub_opt
                    pred_pts_sub_non_overlap_opt = pred_pts_sub_opt
                    frames_sub_non_overlap = frames_sub
                else:
                    labels_gt_sub_non_overlap_opt = labels_gt_sub_opt[:,1:,...]
                    pred_pts_sub_non_overlap_opt = pred_pts_sub_opt[:,1:,...]
                    frames_sub_non_overlap = frames_sub[:,1:,...]

                pred_pts_warped_sub_opt,common_volume = self.intepolation_and_registration_for_each_patch(labels_gt_sub_non_overlap_opt,pred_pts_sub_non_overlap_opt,frames_sub_non_overlap,option = based_volume)
                # convert it into origial coordinates system
                pred_pts_warped_sub_pro = self.convert_from_optimised_to_origin(pred_pts_warped_sub_opt,minxyz_all,convR_batched,labels_gt_sub_non_overlap_opt,common_volume,option = based_volume)
                pred_pts_warped_sub = pred_pts_warped_sub_pro.permute(0,1,3,2)[:,:,0:3,:]
                

            # warped_pred_sub = self.warp_pred(pred_pts_sub,labels_gt_sub,frames_sub)
            # warped_pred_sub = torch.squeeze(warped_pred_sub,0)
            if idx ==0:
                # points in original coordinates system
                labels_gt = labels_gt_sub
                pred_pts = pred_pts_sub
                pred_pts_warped = pred_pts_warped_sub

                labels_gt_local = labels_gt_sub_local 
                pred_pts_local = pred_pts_sub_local 

                ori_pts = ori_pts_sub
                pre = pre_sub
                pred_warped = pred_pts_warped_sub_pro

                ori_pts_local = ori_pts_sub_local
                pre_local = pre_sub_local

                
            else:
                labels_gt = torch.cat((labels_gt,labels_gt_sub[:,1:,...]),1)
                pred_pts = torch.cat((pred_pts,pred_pts_sub[:,1:,...]),1)
                pred_pts_warped = torch.cat((pred_pts_warped,pred_pts_warped_sub),1)

                labels_gt_local = torch.cat((labels_gt_local,labels_gt_sub_local[:,1:,...]),1)
                pred_pts_local = torch.cat((pred_pts_local,pred_pts_sub_local[:,1:,...]),1)

                ori_pts = torch.cat((ori_pts,ori_pts_sub[:,1:,...]),1)
                pre = torch.cat((pre,pre_sub[:,1:,...]),1)

                ori_pts_local = torch.cat((ori_pts_local,ori_pts_sub_local[:,1:,...]),1)
                pre_local = torch.cat((pre_local,pre_sub_local[:,1:,...]),1)

                pred_warped = torch.cat((pred_warped,pred_pts_warped_sub_pro),1)

            if self.option == 'generate_reg_volume_data':
                idx += 1
            elif self.option == "reconstruction_vlume":
                idx += (self.opt.NUM_SAMPLES-1)

        # compute evaluation metrix, given ground truth points, predicted points, and warpped points
        # global transformation on all points
        
        # # compute common frame for comparision with other methods
        # common_idx = (int(frames.shape[1]/self.opt.NUM_SAMPLES)-1)*self.opt.NUM_SAMPLES-int(frames.shape[1]/self.opt.NUM_SAMPLES)+1+1

        # the following is not need as the sequence start from the second sequence
        # pred_pts = pred_pts[0,100:,...][None,...]
        # labels_gt = labels_gt[0,100:,...][None,...]
        # pred_pts_warped = pred_pts_warped[0,100:,...][None,...]
        # pred_pts_local = pred_pts_local[0,100:,...][None,...]
        # labels_gt_local = labels_gt_local[0,100:,...][None,...]

        T_global_all_dist = ((pred_pts-labels_gt)**2).sum(dim=2).sqrt().mean().item()
        # self.metrics(pred_pts,labels_gt).item()
        T_R_wrap_global_all_dist = ((pred_pts_warped-labels_gt)**2).sum(dim=2).sqrt().mean().item()
        # self.metrics(pred_pts_warped,labels_gt).item()
        T_local_all_dist = ((pred_pts_local-labels_gt_local)**2).sum(dim=2).sqrt().mean().item()

        # global transoformation on four corner points
        # pred_pts_four = pred_pts[...,[0,self.dset[0][0].shape[2],(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2]+1,-1]]
        # labels_gt_four = labels_gt[...,[0,self.dset[0][0].shape[2],(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2]+1,-1]]
        
        pred_pts_four = pred_pts[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        labels_gt_four = labels_gt[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        pred_pts_warped_four = pred_pts_warped[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        # pred_pts_four_o = pred_pts_four[0,[99-1,198-1,297-1,396-1,495-1],...]
        pred_pts_four_local = pred_pts_local[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        labels_gt_four_local = labels_gt_local[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        
        # the following is not need as the sequence start from the second sequence
        # pred_pts_four = pred_pts_four[0,100:,...][None,...]
        # labels_gt_four = labels_gt_four[0,100:,...][None,...]
        # pred_pts_warped_four = pred_pts_warped_four[0,100:,...][None,...]
        # pred_pts_four_local = pred_pts_four_local[0,100:,...][None,...]
        # labels_gt_four_local = labels_gt_four_local[0,100:,...][None,...]


        T_global_four_dist = ((pred_pts_four-labels_gt_four)**2).sum(dim=2).sqrt().mean().item()
        T_R_wrap_global_four_dist = ((pred_pts_warped_four-labels_gt_four)**2).sum(dim=2).sqrt().mean().item()
        T_local_four_dist = ((pred_pts_four_local-labels_gt_four_local)**2).sum(dim=2).sqrt().mean().item()
        # global difference on all points
        self.T_Global_AllPts_Dist.append(T_global_all_dist)
        self.T_R_Warp_Global_AllPts_Dist.append(T_R_wrap_global_all_dist)
        self.T_Local_AllPts_Dist.append(T_local_all_dist)

        # global difference on four corner points
        self.T_Global_FourPts_Dist.append(T_global_four_dist)
        self.T_R_Warp_Global_FourPts_Dist.append(T_R_wrap_global_four_dist)
        self.T_Local_FourPts_Dist.append(T_local_four_dist)


        # # plot trajactory based on four corner points
        self.plot_scan(labels_gt_four,pred_pts_four,frames[:,:labels_gt_four.shape[1],...],saved_folder+'/'+saved_name+'_T_global')
        self.plot_scan(labels_gt_four,pred_pts_warped_four,frames[:,:labels_gt_four.shape[1],...],saved_folder+'/'+saved_name+'_TR_global')
        self.plot_scan(labels_gt_four_local,pred_pts_four_local,frames[:,:labels_gt_four.shape[1],...],saved_folder+'/'+saved_name+'_T_local')

        





                
        

