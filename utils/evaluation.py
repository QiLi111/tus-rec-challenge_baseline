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
        self.labels = h5py.File(os.path.join(self.saved_folder,"labels.h5"),'a')

    def calculate_GT_DDF(self, scan_index):
        # calculate DDF of ground truth - label
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
        
        self.labels.create_dataset('/sub%03d_frames%02d' % (indices[0],indices[1]), labels.shape, dtype=labels.dtype, data=labels)



        

        
        

           


    def calculate_predicted_DDF(self, scan_index,saved_folder):

        frames, tforms, scan_name = self.dset[scan_index]
        frames, tforms = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms])
        tforms_inv = torch.linalg.inv(tforms)
        frames = frames/255
        saved_folder = saved_folder


        data_pairs_all = data_pairs_adjacent(frames.shape[1])
        data_pairs_all=torch.tensor(data_pairs_all)

        transform_label_all = LabelTransform(
            label_type=self.opt.LABEL_TYPE,
            pairs=data_pairs_all,  #
            image_points=self.all_points ,
            in_image_coords=True,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
            )
        tforms_each_frame2frame0_gt_all = transform_label_all(tforms, tforms_inv)
        labels_gt_all = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0_gt_all,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]
        # change into optimised coordinates system
        ori_pts = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0_gt_all,torch.matmul(self.tform_calib,self.all_points))).permute(0,1,3,2)
        
        # scale = torch.diag(torch.tensor([0.22447395,0.23554039,1,1])).to(self.device)
        # labels_gt_all = torch.matmul(tforms_each_frame2frame0_gt_all,torch.matmul(scale,self.all_points)[[2,1,0,3]])[:,:,[2,1,0],...]
        # ori_pts = torch.matmul(tforms_each_frame2frame0_gt_all,torch.matmul(scale,self.all_points)[[2,1,0,3]]).permute(0,1,3,2)

        
        
        pre = torch.zeros_like(ori_pts)

        labels_gt_all_opt, pred_pts, convR_batched,minxyz_all = self.ConvPose(labels_gt_all, ori_pts, pre, 'auto_PCA',self.device)
                   
        # # intepolete
        # time1=time.time()
        gt_volume_all_opt, gt_volume_position_all = interpolation_3D_pytorch_batched(scatter_pts = labels_gt_all_opt,
                                                            frames = frames,
                                                            time_log=None,
                                                            saved_folder_test = None,
                                                            scan_name='gt',
                                                            device = self.device,
                                                            option = self.opt.intepoletion_method,
                                                            volume_size = self.opt.intepoletion_volume,
                                                            volume_position = None
                                                            )
        # time2=time.time()-time1
        
        gt_volume_all_ori, gt_volume_position_all = interpolation_3D_pytorch_batched(scatter_pts = labels_gt_all,
                                                            frames = frames,
                                                            time_log=None,
                                                            saved_folder_test = None,
                                                            scan_name='gt',
                                                            device = self.device,
                                                            option = self.opt.intepoletion_method,
                                                            volume_size = self.opt.intepoletion_volume,
                                                            volume_position = None
                                                            )
        
        # save

        save2mha(gt_volume_all_opt[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + str(scan_index)+ '_gt_volume_all_opt.mha'
                )
        
        save2mha(gt_volume_all_ori[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + str(scan_index)+ '_gt_volume_all_ori.mha'
                )


        # plot scan trajactory
        # plot_scan



        idx = 0

        

         
        while True:
            
            
            # start from the second sequence, for comparision with other methods
            
            # in order to let all frames are all based on frames, 
            # two sub-sequences should at least have one frame overlap 
            # for example, 0-99, 99-198, 198-297
            
            

            if (idx + self.opt.NUM_SAMPLES) > frames.shape[1]:
                break

            frames_sub = frames[:,idx:idx + self.opt.NUM_SAMPLES, ...]
            tforms_sub = tforms[:,idx:idx + self.opt.NUM_SAMPLES, ...]
            tforms_inv_sub = tforms_inv[:,idx:idx + self.opt.NUM_SAMPLES, ...]

            # obtain the transformation from current frame to frame 0
            tforms_each_frame2frame0_gt_sub = self.transform_label(tforms_sub, tforms_inv_sub)
            
            # calculate local tarsformation, the previous frame is ground truth
            transf_0 = tforms_each_frame2frame0_gt_sub[:,0:-1,...]
            transf_1 = tforms_each_frame2frame0_gt_sub[:,1:,...]
            tforms_each_frame2frame0_gt_sub_local = torch.matmul(torch.linalg.inv(transf_0),transf_1)
            tforms_each_frame2frame0_gt_sub_local = torch.matmul(transf_0,tforms_each_frame2frame0_gt_sub_local)
            tforms_each_frame2frame0_gt_sub_local = torch.cat((tforms_each_frame2frame0_gt_sub[:,0,...][None,...],tforms_each_frame2frame0_gt_sub_local),1)
            
            with torch.no_grad():
                outputs = self.model(frames_sub)
                # 6 parameter to 4*4 transformation
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

        





                
        # for visulise use
        generate_mha = False
        if generate_mha:
            # change labels to a convenient coordinates system
            if self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord == 'pro_coord':
                # change the corrdinates system of the origial data (label), such that the occupied volume is smallest
                # note: only calculate the nre coordinates syatem of the groundtruth, and predicted points will use the same transformation as groundtruth 
                
                labels_gt_opt, pred_pts_opt,convR_batched,minxyz_all = self.ConvPose(labels_gt, ori_pts, pre, 'auto_PCA',self.device)
                labels_gt_opt1, pred_pts_warped_opt,convR_batched,minxyz_all = self.ConvPose(labels_gt, ori_pts, pred_warped, 'auto_PCA',self.device)
                
                # check the correctness of the code
                if not torch.all(labels_gt_opt==labels_gt_opt1):
                    raise('transformation from original coordinates system to optimised coordinates system is not correct')
            elif self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord != 'pro_coord':
                raise('optimised_coord must be used when pro_coord')

            # generate 

            # intepolete
            # compute the common volume for ground truth and prediction to intepolete
            # min_x,max_x = torch.min(torch.min(labels_gt[0,:,0,:],pred_pts[0,:,0,:])),torch.max(torch.max(labels_gt[0,:,0,:],pred_pts[0,:,0,:]))
            # min_y,max_y = torch.min(torch.min(labels_gt[0,:,1,:],pred_pts[0,:,1,:])),torch.max(torch.max(labels_gt[0,:,1,:],pred_pts[0,:,1,:]))
            # min_z,max_z = torch.min(torch.min(labels_gt[0,:,2,:],pred_pts[0,:,2,:])),torch.max(torch.max(labels_gt[0,:,2,:],pred_pts[0,:,2,:]))

            min_x = torch.min(torch.min(torch.min(labels_gt_opt[0,:,0,:]),torch.min(pred_pts_opt[0,:,0,:])),torch.min(pred_pts_warped_opt[0,:,0,:]))
            max_x = torch.max(torch.max(torch.max(labels_gt_opt[0,:,0,:]),torch.max(pred_pts_opt[0,:,0,:])),torch.max(pred_pts_warped_opt[0,:,0,:]))

            min_y = torch.min(torch.min(torch.min(labels_gt_opt[0,:,1,:]),torch.min(pred_pts_opt[0,:,1,:])),torch.min(pred_pts_warped_opt[0,:,1,:]))
            max_y = torch.max(torch.max(torch.max(labels_gt_opt[0,:,1,:]),torch.max(pred_pts_opt[0,:,1,:])),torch.max(pred_pts_warped_opt[0,:,1,:]))

            min_z = torch.min(torch.min(torch.min(labels_gt_opt[0,:,2,:]),torch.min(pred_pts_opt[0,:,2,:])),torch.min(pred_pts_warped_opt[0,:,2,:]))
            max_z = torch.max(torch.max(torch.max(labels_gt_opt[0,:,2,:]),torch.max(pred_pts_opt[0,:,2,:])),torch.max(pred_pts_warped_opt[0,:,2,:]))




            x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
            y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
            z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
            X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            X, Y, Z =X.to(self.device), Y.to(self.device), Z.to(self.device) 
            common_volume = [X,Y,Z]
            
            gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt_opt,
                                                frames = frames[0,0:labels_gt_opt.shape[1],...],
                                                time_log=None,
                                                saved_folder_test = saved_folder,
                                                scan_name=saved_name+'_gt',
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_position = common_volume,
                                                volume_size = self.opt.intepoletion_volume,

                                                )
            
            pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_opt,
                                                    frames = frames[0,0:pred_pts_opt.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name=saved_name+'_pred',
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
            
            pred_volume_warp,pred_volume_position_warp = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_warped_opt,
                                                    frames = frames[0,0:pred_pts_opt.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name=saved_name+'_pred',
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
            
            print('done')
            # warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
            #             fixed = torch.unsqueeze(gt_volume, 1))

            
            save2mha(gt_volume[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_gt-test.mha'
                )

            save2mha(pred_volume[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_pred-test.mha'
                )
            
            save2mha(pred_volume_warp[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_wrapped-test.mha'
                )


            # generate loval transform-based volume

            # change labels to a convenient coordinates system
            if self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord == 'pro_coord':
                # change the corrdinates system of the origial data (label), such that the occupied volume is smallest
                # note: only calculate the nre coordinates syatem of the groundtruth, and predicted points will use the same transformation as groundtruth 
                
                labels_gt_opt_local, pred_pts_opt_local,convR_batched_local,minxyz_all_local = self.ConvPose(labels_gt_local, ori_pts_local, pre_local, 'auto_PCA',self.device)
                
            elif self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord != 'pro_coord':
                raise('optimised_coord must be used when pro_coord')

            # generate 

            
            min_x = torch.min(torch.min(labels_gt_opt_local[0,:,0,:]),torch.min(pred_pts_opt_local[0,:,0,:]))
            max_x = torch.max(torch.max(labels_gt_opt_local[0,:,0,:]),torch.max(pred_pts_opt_local[0,:,0,:]))

            min_y = torch.min(torch.min(labels_gt_opt_local[0,:,1,:]),torch.min(pred_pts_opt_local[0,:,1,:]))
            max_y = torch.max(torch.max(labels_gt_opt_local[0,:,1,:]),torch.max(pred_pts_opt_local[0,:,1,:]))

            min_z = torch.min(torch.min(labels_gt_opt_local[0,:,2,:]),torch.min(pred_pts_opt_local[0,:,2,:]))
            max_z = torch.max(torch.max(labels_gt_opt_local[0,:,2,:]),torch.max(pred_pts_opt_local[0,:,2,:]))




            x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
            y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
            z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
            X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            X, Y, Z =X.to(self.device), Y.to(self.device), Z.to(self.device) 
            common_volume = [X,Y,Z]
            
            gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt_opt_local,
                                                frames = frames[0,0:labels_gt_opt_local.shape[1],...],
                                                time_log=None,
                                                saved_folder_test = saved_folder,
                                                scan_name=saved_name+'_gt',
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_position = common_volume,
                                                volume_size = self.opt.intepoletion_volume,

                                                )
            
            pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_opt_local,
                                                    frames = frames[0,0:pred_pts_opt_local.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name=saved_name+'_pred',
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
            
            
            
            print('done')
            # warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
            #             fixed = torch.unsqueeze(gt_volume, 1))

            
            save2mha(gt_volume[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_gt-test-local.mha'
                )

            save2mha(pred_volume[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_pred-test-local.mha'
                )
            
           


        


        # print('done')

        
       
        
    
    def generate_ddf(self,frames,labels_gt,pred_pts):
        # generate ddf from points in optimised coordinates system
        # as the trained registartion network is a smaller volume which only include 100 frames,
        # it is better to generate ddf every 100 frames
        
        idx = 0
         
        while True:
            if (idx + self.opt.NUM_SAMPLES) > labels_gt.shape[1]:
                break
            # pick up sequence US similar as the training set size
            labels_gt_seq = labels_gt[:,idx:idx + self.opt.NUM_SAMPLES, ...]
            pred_pts_seq = pred_pts[:,idx:idx + self.opt.NUM_SAMPLES, ...]

            frames_seq = frames[:,idx:idx + self.opt.NUM_SAMPLES, ...]


            # get the ddf, which is the wrapped fixed if self.opt.ddf_dirc == 'Move'
            if self.opt.ddf_dirc == 'Move':
                # obtain wrapped fixed and the ddf which is based on the moving/prediction
                pred_pts_warped = self.intepolation_and_registration_for_each_patch(labels_gt_seq,pred_pts_seq,frames_seq,based_volume)

                warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_pts_seq, 1), 
                        fixed = torch.unsqueeze(labels_gt_seq, 1))
                


            elif self.opt.ddf_dirc == 'Fix':
                raise('TBC')



            idx += (self.opt.NUM_SAMPLES-1)

    def intepolation_and_registration_for_each_patch(self,labels_gt,pred_pts,frames,option):
        # use common volme to generate volume from scatter points
        if option == 'common_volume':
            # compute the common volume for ground truth and prediction to intepolete
            min_x,max_x = torch.min(torch.min(labels_gt[0,:,0,:],pred_pts[0,:,0,:])),torch.max(torch.max(labels_gt[0,:,0,:],pred_pts[0,:,0,:]))
            min_y,max_y = torch.min(torch.min(labels_gt[0,:,1,:],pred_pts[0,:,1,:])),torch.max(torch.max(labels_gt[0,:,1,:],pred_pts[0,:,1,:]))
            min_z,max_z = torch.min(torch.min(labels_gt[0,:,2,:],pred_pts[0,:,2,:])),torch.max(torch.max(labels_gt[0,:,2,:],pred_pts[0,:,2,:]))


            x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
            y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
            z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
            X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            X, Y, Z =X.to(self.device), Y.to(self.device), Z.to(self.device) 
            common_volume = [X,Y,Z]
            
            gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt,
                                                frames = frames,#[0,0:labels_gt.shape[1],...],
                                                time_log=None,
                                                saved_folder_test = None,
                                                scan_name=None,
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_position = common_volume,
                                                volume_size = self.opt.intepoletion_volume,

                                                )
            
            pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts,
                                                    frames = frames,#[0,0:pred_pts.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = None,
                                                    scan_name=None,
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
        else:
            raise('TBC')

        if self.opt.ddf_dirc == 'Move':
            # generate wrapped fixed
            warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
                            fixed = torch.unsqueeze(gt_volume, 1))
            
            pred_pts_warped = self.generate_wraped_prediction(warped, ddf,pred_volume,pred_pts,labels_gt,common_volume,option = option)
        
        else:
            pred_pts_warped = pred_pts
            
        
        return pred_pts_warped,common_volume


    def warp_pred(self,pred_pts,labels_gt,frames):
        # get common volume


        
        gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt,
                                            frames = frames,
                                            time_log=None,
                                            saved_folder_test = None,
                                            scan_name=None,
                                            device = self.device,
                                            option = self.opt.intepoletion_method,
                                            volume_position = None,
                                            volume_size = self.opt.intepoletion_volume,

                                            )
        
        pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts,
                                                   frames = frames,
                                                   time_log=None,
                                                   saved_folder_test = None,
                                                   scan_name=None,
                                                   device = self.device,
                                                   option = self.opt.intepoletion_method,
                                                   volume_position = gt_volume_position,
                                                   volume_size = self.opt.intepoletion_volume,
                                                 )

        warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
                    fixed = torch.unsqueeze(gt_volume, 1))
        return warped

        
    # def compute_location_diffence(self):


    def generate_wraped_prediction(self,warped, ddf,pred_volume,pred_pts,gt_pts,common_volume,option):
        # generate warped moving (prediction) from DDF (DDF is the displacement on moving image)
        
        # intopolate into scatter DDF from grid DDF
        # ddf_scatter = torch.zeros(ddf.shape[0],ddf.shape[1],pred_pts.shape)
        # reshapa and permute to satisfy grid_sample requirment 
        pred_pts = torch.reshape(pred_pts, (pred_pts.shape[0],pred_pts.shape[1],pred_pts.shape[2],self.dset[0][0].shape[1],self.dset[0][0].shape[2])).permute(0,1,3,4,2)
        
        #  normalized into [0,N], which has no unite, float type
        # as DDF is in range [0,N], which is defined by voxelMorph, such that prediction points locations should be 
        # normalised into the same space, and then ADD operation can be added
        pred_pts_norm = torch.zeros_like(pred_pts)
        pred_pts_0_1 = torch.zeros_like(pred_pts)
        
        min_x,max_x = torch.min(common_volume[0]),torch.max(common_volume[0])
        min_y,max_y = torch.min(common_volume[1]),torch.max(common_volume[1])
        min_z,max_z = torch.min(common_volume[2]),torch.max(common_volume[2])
        minxyz = torch.from_numpy(np.array([min_x.item(),min_y.item(),min_z.item()]))

        
        for i in range(pred_pts.shape[-1]):
            # gt_pts is in convinient system
            # the operatipon of normalise of prediction points should be exactly the same as the operation for intepolation for
            # ground truth points and prediction points
            if option == 'gt_based_volume':
                pred_pts_norm[...,i] = (pred_pts[...,i]-torch.min(gt_pts[:,:,i,:]))/1 # 1 is the spaceing, which should be consistent with the value in intepolation function
            elif option == 'common_volume':
                
                pred_pts_norm[...,i] = (pred_pts[...,i]-minxyz[i])/1 # 1 is the spaceing, which should be consistent with the value in intepolation function

            else:
                raise('TBC')
        # normalise pred_pts into [-1,1] to satisfy the grid_sample requirment
        for i, dim in enumerate(ddf.shape[2:]):#[-1:-4:-1]
            pred_pts_0_1[..., 2-i] = pred_pts_norm[..., i] * 2 / (dim - 1) - 1

        # pred_pts_0_1 = torch.flip(pred_pts_0_1, -1)
        # pred_pts_0_1 = pred_pts_0_1.permute(0,2,3,1,4)

        # for i in range(ddf_scatter.shape[1]):
        # ddf = torch.zeros_like(ddf)+20 #for test

        # ddf[:,0,...]=torch.zeros_like(ddf[:,0,...])+20
        # ddf[:,1,...]=torch.zeros_like(ddf[:,1,...])
        # ddf[:,2,...]=torch.zeros_like(ddf[:,2,...])

        ddf_scatter = torch.nn.functional.grid_sample(
                                                input = ddf, 
                                                grid = pred_pts_0_1, 
                                                mode='bilinear', 
                                                padding_mode='zeros', align_corners=False)
        

        # generate warpped moving/prediction image
        ddf_scatter = ddf_scatter.permute(0,2,3,4,1)
        # ddf_scatter = ddf_scatter.permute(0,4,2,3,1)
        pred_pts_warped = pred_pts_norm+ddf_scatter
        pred_pts_warped = pred_pts_warped.permute(0,1,4,2,3)
        pred_pts_warped = torch.reshape(pred_pts_warped,(pred_pts_warped.shape[0],pred_pts_warped.shape[1],pred_pts_warped.shape[2],-1))


        return pred_pts_warped


    def calculateConvPose_batched(self,pts_batched,option,device):
        for i_batch in range(pts_batched.shape[0]):
        
            ConvR = self.calculateConvPose(pts_batched[i_batch,...],option,device)
            # ConvR = ConvR.repeat(pts_batched[i_batch,...].shape[0], 1,1)[None,...]
            ConvR = ConvR[None,...]
            if i_batch == 0:
                ConvR_batched = ConvR
            else:
                ConvR_batched = torch.cat((ConvR_batched,ConvR),0)
            return ConvR_batched
            

    def calculateConvPose_batched1(self,pts_batched,option,device):
        for i_batch in range(pts_batched.shape[0]):
            
            ConvR = self.calculateConvPose(pts_batched[i_batch,...],option,device)
            # ConvR = ConvR.repeat(pts_batched[i_batch,...].shape[0], 1,1)[None,...]
            ConvR = ConvR[None,...]
            if i_batch == 0:
                ConvR_batched = ConvR
            else:
                ConvR_batched = torch.cat((ConvR_batched,ConvR),0)
        return ConvR_batched


    def calculateConvPose(self,pts,option,device):
        """Calculate roto-translation matrix from global reference frame to *convenient* reference frame.
        Voxel-array dimensions are calculated in this new refence frame. This rotation is important whenever the US scans sihouette is remarkably
        oblique to some axis of the global reference frame. In this case, the voxel-array dimensions (calculated by the smallest parallelepipedon 
        wrapping all the realigned scans), calculated in the global refrence frame, would not be optimal, i.e. larger than necessary.
        
        .. image:: diag_scan_direction.png
            :scale: 30 %          
            
        Parameters
        ----------
        convR : mixed
            Roto-translation matrix.
            If str, it specifies the method for automatically calculate the matrix.
            If 'auto_PCA', PCA is performed on all US image corners. The x, y and z of the new convenient reference frame are represented by the eigenvectors out of the PCA.
            If 'first_last_frames_centroid', the convenent reference frame is expressed as:
            
            - x from first image centroid to last image centroid
            - z orthogonal to x and the axis and the vector joining the top-left corner to the top-right corner of the first image
            - y orthogonal to z and x
            
            If np.ndarray, it must be manually specified as a 4 x 4 affine matrix.
            
        """
        # pts = torch.reshape(pts,(pts.shape[0],-1,3))
        # pts = torch.permute(pts, (2, 0, 1))
        
        # Calculating best pose automatically, if necessary
        # ivx = np.array(self.voxFrames)
        if option == 'auto_PCA':
            # Perform PCA on image corners
            # print ('Performing PCA on images corners...')
            with torch.no_grad():
                pts1 = pts.permute(0,2,1).reshape([-1,3])#.cpu().numpy()
                U, s = self.pca(torch.transpose(pts1, 0, 1)) 
                # Build convenience affine matrix
                convR = torch.vstack((torch.hstack((U,torch.zeros((3,1)).to(device))),torch.tensor([0,0,0,1]).to(device)))#.T
                # convR = torch.from_numpy(convR).to(torch.float32).to(device)
            # print ('PCA perfomed')
        elif option == 'first_last_frames_centroid':
            # Search connection from first image centroid to last image centroid (X)
            # print ('Performing convenient reference frame calculation based on first and last image centroids...')
            C0 = torch.mean(pts[0,:,:], 1)  # 3
            C1 = torch.mean(pts[-1,:,:], 1)  # 3
            X = C1 - C0
            # Define Y and Z axis
            Ytemp = pts[0,:,0] - pts[0,:,1]   # from top-left corner to top-right corner
            
            Z = torch.cross(X, Ytemp)
            Y = torch.cross(Z, X)
            # Normalize axis length
            X = X / torch.linalg.norm(X)
            Y = Y / torch.linalg.norm(Y)
            Z = Z / torch.linalg.norm(Z)
            # Create rotation matrix
            # M = np.array([X, Y, Z]).T
            M = torch.transpose(torch.stack((X,Y,Z),0),0,1)
            # Build convenience affine matrix
            # convR = np.vstack((np.hstack((M,np.zeros((3,1)))),[0,0,0,1])).T
            convR = torch.transpose(torch.vstack((torch.hstack((M,torch.zeros((3,1)).to(device))),torch.tensor([0,0,0,1]).to(device))),0,1)
            # print ('Convenient reference frame calculated')

        return convR

    def ConvPose(self,labels,ori_pts,pre, option_method,device):
        convR_batched = self.calculateConvPose_batched(labels,option = option_method,device=device)    
        
        for i_batch in range(convR_batched.shape[0]):
            
            labels_i = torch.matmul(ori_pts[i_batch,...],convR_batched[i_batch,...])[None,...]
            minx = torch.min(labels_i[...,0])
            miny = torch.min(labels_i[...,1])
            minz = torch.min(labels_i[...,2])
            labels_i[...,0]-=minx
            labels_i[...,1]-=miny
            labels_i[...,2]-=minz

            pred_pts_i = torch.matmul(pre[i_batch,...],convR_batched[i_batch,...])[None,...]
            
            pred_pts_i[...,0]-=minx
            pred_pts_i[...,1]-=miny
            pred_pts_i[...,2]-=minz

            # return for future use
            minxyz=torch.from_numpy(np.array([minx.item(),miny.item(),minz.item()]))

            if i_batch == 0:
                labels_opt = labels_i
                pred_pts_opt = pred_pts_i
                minxyz_all = minxyz
            else:
                labels_opt = torch.cat((labels_opt,labels_i),0)
                pred_pts_opt = torch.cat((pred_pts_opt,pred_pts_i),0)
                minxyz_all = torch.cat((minxyz_all,minxyz),0)

        labels_opt = labels_opt[:,:,:,0:3].permute(0,1,3,2)
        pred_pts_opt = pred_pts_opt[:,:,:,0:3].permute(0,1,3,2)
        
        #only for debug 2024.02.08

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x=labels_opt[...,0,:].reshape(-1)[::1000].cpu().numpy()
# y=labels_opt[...,1,:].reshape(-1)[::1000].cpu().numpy()
# z=labels_opt[...,2,:].reshape(-1)[::1000].cpu().numpy()
# ax.scatter(x,y,z,color='y',alpha=0.2)
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(x,y,color='y',alpha=0.2)
# plt.savefig('./test.png')
# x=labels[...,0,:].reshape(-1)[::1000].cpu().numpy()
# y=labels[...,1,:].reshape(-1)[::1000].cpu().numpy()
# z=labels[...,2,:].reshape(-1)[::1000].cpu().numpy()
# ax.scatter(x,y,z,color='y',alpha=0.2)
# center=np.mean(np.stack([x,y,z],axis=0),axis=1)
# u1=center+convR_batched[0,:3,0].cpu().numpy()*50
# u2=center+convR_batched[0,:3,1].cpu().numpy()*50
# u3=center+convR_batched[0,:3,2].cpu().numpy()*50
# ax.plot([center[0],u1[0]],[center[1],u1[1]],[center[2],u1[2]],color='r')
# ax.plot([center[0],u2[0]],[center[1],u2[1]],[center[2],u2[2]],color='r')
# ax.plot([center[0],u3[0]],[center[1],u3[1]],[center[2],u3[2]],color='r')
# plt.savefig('./test.png')
        # plt.show()
        
        
        return labels_opt,pred_pts_opt,convR_batched,minxyz_all

    def pca(self,D):
        """Run Principal Component Analysis on data matrix. It performs SVD
        decomposition on data covariance matrix.
        
        Parameters
        ----------
        D : np.ndarray
            Nv x No matrix, where Nv is the number of variables 
            and No the number of observations.
        
        Returns
        -------
        list
            U, s as out of SVD (``see np.linalg.svd``)

        """
        cov = torch.cov(D)
        U, s, V = torch.linalg.svd(cov)
        return U, s

    def convert_from_optimised_to_origin(self,pred_pts_warped_sub,minxyz_all,convR_batched,labels_gt_sub_opt,common_volume,option = 'common_volume'):
        
        

        if option == 'gt_volume':
        
            for i in range(pred_pts_warped_sub.shape[2]):
                pred_pts_warped_sub[:,:,i,:] = pred_pts_warped_sub[:,:,i,:]*1+torch.min(labels_gt_sub_opt[:,:,i,:]) # 1 is the spaceing, which should be consistent with the value in intepolation function
        elif option == 'common_volume':
            min_x,max_x = torch.min(common_volume[0]),torch.max(common_volume[0])
            min_y,max_y = torch.min(common_volume[1]),torch.max(common_volume[1])
            min_z,max_z = torch.min(common_volume[2]),torch.max(common_volume[2])
            minxyz = torch.from_numpy(np.array([min_x.item(),min_y.item(),min_z.item()]))



            for i in range(pred_pts_warped_sub.shape[2]):
                pred_pts_warped_sub[:,:,i,:] = pred_pts_warped_sub[:,:,i,:]*1+minxyz[i] # 1 is the spaceing, which should be consistent with the value in intepolation function
        
        else:
            raise('TBC')

        
        if pred_pts_warped_sub.shape[0]==1:
            pred_pts_warped_sub[:,:,0,:]+=minxyz_all[0]
            pred_pts_warped_sub[:,:,1,:]+=minxyz_all[1]
            pred_pts_warped_sub[:,:,2,:]+=minxyz_all[2]

            pred_pts_warped_sub = pred_pts_warped_sub.permute(0,1,3,2)
            pred_pts_warped_sub_pad = F.pad(input=pred_pts_warped_sub, pad=(0, 1, 0, 0), mode='constant', value=1)
            pred_pts_warped_sub_ori = torch.matmul(pred_pts_warped_sub_pad,torch.linalg.inv(convR_batched))

        elif pred_pts_warped_sub.shape[0]>1:
            raise('batched not implemented')

        return pred_pts_warped_sub_ori
        

    def plot_scan(self,labels_gt_four,pred_pts_four,frames,saved_name):

        # save numpy file, for uture plot use
        all_frames_fd = '/'+'/'.join(saved_name.split('/')[1:7])+'/frames_in_testset'
        if not os.path.exists(all_frames_fd):
            os.makedirs(all_frames_fd)

        all_gt_fd = '/'+'/'.join(saved_name.split('/')[1:7])+'/gts_in_testset'
        if not os.path.exists(all_gt_fd):
            os.makedirs(all_gt_fd)
    

        with open(all_gt_fd+'/'+saved_name.split('/')[-1]+'_gt_noncommon.npy', 'wb') as f:
            np.save(f, labels_gt_four.cpu().numpy())
        with open(saved_name+'_pred_noncommon.npy', 'wb') as f:
            np.save(f, pred_pts_four.cpu().numpy())
        with open(all_frames_fd+'/'+saved_name.split('/')[-1]+'_frame_noncommon.npy', 'wb') as f:
            np.save(f, frames.cpu().numpy())

        ax = plt.figure().add_subplot(projection='3d')
        # plot the frame 0
        # px, py, pz = [torch.mm(self.tform_calib_scale, torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.all_points))[ii,].reshape(2, 2) for ii in range(3)]
        # pix_intensities = (frames[0, ..., None].float() / 255).cpu().expand(-1, -1, 3).numpy()
        # fx, fy, fz = [torch.mm(self.tform_calib_scale.cpu(), torch.mm(torch.from_numpy(np.array([[self.opt.RESAMPLE_FACTOR, 0, 0, 0], [0, self.opt.RESAMPLE_FACTOR, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],np.float32)), self.pixel_points.cpu()))[ii,].reshape(2, 2) for ii in range(3)]

        if labels_gt_four.shape[0]==1:
            fx, fy, fz = [labels_gt_four[:,0,...].cpu().numpy()[:,ii,].reshape(2, 2) for ii in range(3)]
            pix_intensities = (frames[0,0, ..., None].float() / 255).expand(-1, -1, 3).cpu().numpy()
        else:
            raise('the first dimention must be 1')

        ax.plot_surface(fx, fy, fz, facecolors=pix_intensities, linewidth=0, antialiased=True)
        ax.plot_surface(fx, fy, fz, edgecolor='g', linewidth=1, alpha=0.2, antialiased=True)

        gx_all, gy_all, gz_all = [labels_gt_four[:, :, ii, :].cpu().numpy() for ii in range(3)]
        prex_all, prey_all, prez_all = [pred_pts_four[:, :, ii, :].cpu().numpy() for ii in range(3)]
        ax.scatter(gx_all, gy_all, gz_all, c='g', alpha=0.2, s=2)
        ax.scatter(prex_all, prey_all, prez_all, c='r', alpha=0.2, s=2)
        # for i in range(4):
        #     ax.plot(prex_all[0,:,i], prey_all[0,:,i], prez_all[0,:,i],'.-', c='r',linewidth = 1)



        # prex_all_o, prey_all_o, prez_all_o = [pred_pts_four_o[:, ii, :].cpu().numpy() for ii in range(3)]
        # ax.scatter(prex_all_o, prey_all_o, prez_all_o, 'b^', s=5)

        # plot the last image
        gx, gy, gz = [labels_gt_four[:, -1, ii, :].cpu().numpy() for ii in range(3)]
        prex, prey, prez = [pred_pts_four[:, -1, ii, :].cpu().numpy() for ii in range(3)]


        gx, gy, gz = gx.reshape(2, 2), gy.reshape(2, 2), gz.reshape(2, 2)
        ax.plot_surface(gx, gy, gz, edgecolor='g', linewidth=1, alpha=0.2, antialiased=True, label='gt')#
        prex, prey, prez = prex.reshape(2, 2), prey.reshape(2, 2), prez.reshape(2, 2)
        ax.plot_surface(prex, prey, prez, edgecolor='r', linewidth=1, alpha=0.2, antialiased=True, label='pred')
        ax.axis('equal')
        ax.legend()


        # plt.show()

        plt.savefig(saved_name+'.png')
        plt.savefig(saved_name+'.pdf')

        plt.close()




