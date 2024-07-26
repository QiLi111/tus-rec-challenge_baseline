import torch,os
from matplotlib import pyplot as plt
import numpy as np


def plot_scans(frames,tforms,scan_name,label_GP,pred_GP,saved_folder,tform_calib_scale,image_points):
    # plot scan in 3D

    # get four corner points
    label_global_four = select4pts(label_GP,tform_calib_scale,image_points,frames)
    pred_global_four = select4pts(pred_GP,tform_calib_scale,image_points,frames)

    frames, tforms = (torch.tensor(t) for t in [frames, tforms])
    frames = frames/255
    
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)

    color = ['g','r']
    # plot label and prediction separately
    plot_scan_individual(label_global_four,frames,os.path.join(saved_folder,'sub%03d__%s' % (int(scan_name.split('__')[0][3:]),scan_name.split('__')[1])+'_label'),step = frames.shape[0]-1,color = color[0],width = 4, scatter = 8, legend_size=50, legend = 'GT')
    plot_scan_individual(pred_global_four,frames,os.path.join(saved_folder,'sub%03d__%s' % (int(scan_name.split('__')[0][3:]),scan_name.split('__')[1])+'_pred'),step = frames.shape[0]-1,color = color[1],width = 4, scatter = 8, legend_size=50, legend = 'Pred')
    # plot label and prediction in the same figure 
    plot_scan_label_pred(label_global_four,pred_global_four,frames,color,os.path.join(saved_folder,'sub%03d__%s' % (int(scan_name.split('__')[0][3:]),scan_name.split('__')[1])+'_pred_label'),step = frames.shape[0]-1,width = 4, scatter = 8, legend_size=50)

def plot_scan_individual(gt,frame,saved_name,step,color,width = 4, scatter = 8, legend_size=50,legend = None):
    # plot the scan in 3D

    fig = plt.figure(figsize=(35,15))
    axs=[]
    for i in range(2):
        axs.append(fig.add_subplot(1,2,i+1,projection='3d'))
    plt.tight_layout()

    plotting(gt,frame,axs,step,color,width, scatter, legend_size,legend = legend)
    plt.savefig(saved_name +'.png')
    plt.close()

def plotting(gt,frame,axs,step,color,width = 4, scatter = 8, legend_size=50,legend = None): 
    # plot surface
    ysize, xsize = frame.shape[-2:]
    grid=np.meshgrid(np.linspace(0,1,ysize),np.linspace(0,1,xsize),indexing='ij')
    coord = np.zeros((3,ysize,xsize))

    for i_frame in range(0,gt.shape[0],step): 
        gx, gy, gz = [gt[i_frame, ii, :] for ii in range(3)]
        gx, gy, gz = gx.reshape(2, 2), gy.reshape(2, 2), gz.reshape(2, 2)
        coord[0]=gx[0,0]+(gx[1,0]-gx[0,0])*(grid[0])+(gx[0,1]-gx[0,0])*(grid[1])
        coord[1]=gy[0,0]+(gy[1,0]-gy[0,0])*(grid[0])+(gy[0,1]-gy[0,0])*(grid[1])
        coord[2]=gz[0,0]+(gz[1,0]-gz[0,0])*(grid[0])+(gz[0,1]-gz[0,0])*(grid[1])
         
        pix_intensities = (frame[i_frame, ...]/frame[i_frame, ...].max())
        for i,ax in enumerate(axs):
            ax.plot_surface(coord[0], coord[1], coord[2], facecolors=plt.cm.gray(pix_intensities), shade=False,linewidth=0, antialiased=True, alpha=0.5)
    # plot gt
    gx_all, gy_all, gz_all = [gt[:, ii, :] for ii in range(3)]
    for i,ax in enumerate(axs):
        ax.scatter(gx_all[...,0], gy_all[...,0], gz_all[...,0],  alpha=0.5, c = color, s=scatter, label=legend)
        ax.scatter(gx_all[...,1], gy_all[...,1], gz_all[...,1],  alpha=0.5,c = color, s=scatter)
        ax.scatter(gx_all[...,2], gy_all[...,2], gz_all[...,2],  alpha=0.5, c = color,s=scatter)
        ax.scatter(gx_all[...,3], gy_all[...,3], gz_all[...,3],  alpha=0.5,c = color, s=scatter)
        # plot the first frame and the last frame
        ax.plot(gt[0,0,0:2], gt[0,1,0:2], gt[0,2,0:2], 'b', linewidth = width)
        ax.plot(gt[0,0,[1,3]], gt[0,1,[1,3]], gt[0,2,[1,3]], 'b', linewidth = width) 
        ax.plot(gt[0,0,[3,2]], gt[0,1,[3,2]], gt[0,2,[3,2]], 'b', linewidth = width) 
        ax.plot(gt[0,0,[2,0]], gt[0,1,[2,0]], gt[0,2,[2,0]], 'b', linewidth = width)
        ax.plot(gt[-1,0,0:2], gt[-1,1,0:2], gt[-1,2,0:2], 'r', linewidth = width)
        ax.plot(gt[-1,0,[1,3]], gt[-1,1,[1,3]], gt[-1,2,[1,3]], 'r', linewidth = width) 
        ax.plot(gt[-1,0,[3,2]], gt[-1,1,[3,2]], gt[-1,2,[3,2]], 'r', linewidth = width) 
        ax.plot(gt[-1,0,[2,0]], gt[-1,1,[2,0]], gt[-1,2,[2,0]], 'r', linewidth = width)


        ax.axis('equal')
        ax.grid(False)
        ax.legend(fontsize = legend_size,markerscale = 5,scatterpoints = 5)
        # ax.axis('off')
        ax.set_xlabel('x',fontsize=legend_size)
        ax.set_ylabel('y',fontsize=legend_size)
        ax.set_zlabel('z',fontsize=legend_size)
        plt.rc('xtick', labelsize=legend_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=legend_size)    # fontsize of the tick labels
        if i==0:
            ax.view_init(10,30,0)
        else:
            ax.view_init(30,30,0)

def plot_scan_label_pred(gt,pred,frame,color,saved_name,step,width = 4, scatter = 8, legend_size=50):
    # plot the scan in 3D

    fig = plt.figure(figsize=(35,15))
    axs=[]
    for i in range(2):
        axs.append(fig.add_subplot(1,2,i+1,projection='3d'))
    plt.tight_layout()

    plotting(gt,frame,axs,step,color[0],width = 4, scatter = 8, legend_size=50,legend = 'GT')
    plotting(pred,frame,axs,step,color[1],width = 4, scatter = 8, legend_size=50,legend = 'Pred')
    
    plt.savefig(saved_name +'.png')
    plt.close()

def select4pts(GP,tform_calib_scale,image_points,frames):
    # index four corner pts for each frame, from all pixels

    global_allpts = GP + torch.matmul(tform_calib_scale,image_points)[0:3,:].expand(GP.shape[0],-1,-1).numpy()
    global_four = global_allpts[...,[0,frames.shape[-1]-1,(frames.shape[-2]-1)*frames.shape[-1],-1]]
    # add the first frame
    first_frame_coord_all = torch.matmul(tform_calib_scale,image_points.cpu())[0:3,:]
    first_frame_coord = first_frame_coord_all.numpy()[...,[0,frames.shape[-1]-1,(frames.shape[-2]-1)*frames.shape[-1],-1]][None,...]
    global_four = np.concatenate((first_frame_coord,global_four),axis = 0)

    return global_four
    
