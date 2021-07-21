#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:30:47 2021

@author: mohammad Ghahramani

It reads a video, 

- imports frame by frame, 
- runs openpose,
- obtains midpoints
- aligns the frame
- runs keypoints to obtain them
- saves the final image as the output


"""


def blazepose_inf(main_path,movie_file,last_frame):

# global main_path
# main_path = 'c:/downloads/'
    movie_file = main_path+movie_file
    
    # addtional vars
    
    saving = True
    saving_aux = False
    
    # importing requirements
    
    import tensorflow as tf
    import os
    import numpy as np
    import cv2
    from scipy import ndimage
    import time
    from PIL import Image
    import matplotlib.patches as patches
    import pandas as pd
    
    
    # saving global settings
    
    
    
    # import pre written utilities for the whole pipeline
    import os
    os.chdir(main_path)
    import aal_bp_utilities 
    import aal_bp_post_proc_utils 
    
    df = pd.DataFrame({'path': [main_path]})
    df.to_csv('out.csv', index=False)  
    
    # import all utils for blazeface
    os.chdir(main_path)
    import bface_utils 
    
    # import all utils for blazepose and trained params 
    os.chdir(main_path)
    import bpose_utils 
    
    # Delete previous results
    aal_bp_utilities.deleteing_prev_results(main_path)
    
    # create folders and analysis
    aal_bp_utilities.create_all_folders(main_path)
    
    # read and instantiate the video file
    vidcap = cv2.VideoCapture(movie_file)
    # get the video length as vid_len so that we do not exceed reading frames
    vid_len=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # setting parameters to read the video frames
    limit=1
    
    # how many seconds to pass from the begining of the video due to ads or still body
    starting_second = 35
    # how many frames do we process after the starting_second
    # last_frame=100
    vidcap.set(cv2.CAP_PROP_POS_MSEC, starting_second * 1000)
    success,image = vidcap.read()
    
    # save one frame from the video to extract image parameters such as size, ... later in the code
    ref_image=image
    im_size0 = ref_image.shape[0]
    im_size1 = ref_image.shape[1]
    
    
    height, width, layers = image.shape
    ref_height, ref_width, layers = ref_image.shape
    
    frameSize = (width, height)
    
    aa = time.time()
    # save annotated output video after face detection if saving is True
    out_ann , path_saved = aal_bp_utilities.make_outvideo_and_path('face_out_annotated.mp4','vid_test/temp/main/main_face/',main_path,frameSize)
    print("Performing blazeface on the video")
    vect_dic,crop_dic0,crop_dic_dir,main_face,main_face_II,main_face0 = aal_bp_utilities.apply_blazeface(vid_len,vidcap,last_frame,image,success,bface_utils.net,out_ann,main_path,saving)
    
    
   
    # read images cropped after blazeface
    print("Loading the files for blazepose ...")
    path_saved=main_path+'vid_test/temp/main/face_out0/'
    data = aal_bp_utilities.read_images_after_blazeface(path_saved,vect_dic,main_face0)
    
    # extract keypoints
    y = bpose_utils.model.predict(data)
    
    # postprocess keypoints
    ex_labels = aal_bp_utilities.post_process_keypoints(y)
    shoulder_points = np.copy(ex_labels)
    
    if saving_aux:
        out_ann , path_saved = aal_bp_utilities.make_outvideo_and_path('shoulder_out_annotated.mp4','vid_test/temp/main/main_face/',main_path,frameSize)
        path_saved_shoulder=main_path+'vid_test/temp/main/main_shoulder/'
        aal_bp_utilities.show_shoulder_results(ex_labels,crop_dic0,out_ann,'shoulder_out_annotated.mp4',path_saved,path_saved_shoulder,main_path)
        
    # extract body angle based on head and shoulders, mid shoulder points save them in angle_seq (angle_sequence)
    angle_seq = aal_bp_utilities.calculate_body_angle(ex_labels)
    
    # estimate body angle and cropping region parameters for mid hip detection
    path_saved=main_path+'vid_test/temp/main/main_face_II/'
    # img_mask,rot_params = images_for_midhip(path_saved,angle_seq,crop_dic_dir,main_path)
    
        
    jpegfiles = list()
    # jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    jpegfiles = sorted(main_face_II.keys())
    
     
    
    rot_params={}
    img_mask=np.zeros((2,angle_seq.shape[0]))
    
    # a1 = time.time()
    for t in range(len(jpegfiles)):
        II=main_face_II[jpegfiles[t]] #cv2.imread(path_saved+jpegfiles[t])
        # II=cv2.imread(path_saved+jpegfiles[t])

        ref_box=crop_dic_dir[str(t)][0].reshape(2,2)
        
        # convert zero degree to 30 and 180 degree to 150 that are extreme cosin conditions 
        z=aal_bp_utilities.fix_angle_extreams(angle_seq[t,1])
          
        # an optimized super fast angular croping based on meshing the image taking care of adding padding to the image
        sam,saml,stx,sty,size1,size2,sec_points,im3,stxl,styl,size1l,size2l,sec_pointsl,im3l = bface_utils.angular_cropping(II,ref_box,crop_dic_dir[str(t)][1],crop_dic_dir[str(t)][2],z)
        
        # assigning rotation parameters in a dictionary for reversing the keypoints
        temp_dict, temp_dictl = aal_bp_post_proc_utils.mid_hip_image(sam,saml,stx,sty,size1,size2,sec_points,im3,stxl,styl,size1l,size2l,sec_pointsl,im3l)
        rot_params['im'+str(t).zfill(4)+'_in'+'1']=temp_dict
        rot_params['im'+str(t).zfill(4)+'_in'+'2']=temp_dictl
        
        # taking care of short and long people improvement to blazepose
        file_name = main_path+'vid_test/temp/main/face_out_1/'+jpegfiles[t][0:-5]+'_01'+'.jpeg'
        file_namel = main_path+'vid_test/temp/main/face_out_2/'+jpegfiles[t][0:-5]+'_02'+'.jpeg'
    
        # save aligned files for finding the midhip location
        image_mask = aal_bp_post_proc_utils.save_files_for_midhips(file_name,file_namel,z,sam,saml)
        img_mask[:,t]=image_mask.astype(int)        
    # print('wriring',time.time()-a1)
    # implement finding midhips to align images for blazepose
    
    #initiate variables as zero matrices/arrays
    labels=np.zeros((2,angle_seq.shape[0],11,3))
    
    # initiate variables for reversing back the labels from their cropped version to the original version saved in transfered labels
    transfered_labels=np.zeros((2,angle_seq.shape[0],11,3))
    
    # read both images for medium (bendover) and long bodies
    
    for u in range(1,3):
        if u==2:
            # print(u)
            path_saved=main_path+'vid_test/temp/main/face_out_'+str(u)+'/'
            
            # apply the model and 
            y=bpose_utils.read_data_for_midhip_extraction(path_saved,angle_seq,bpose_utils.model_rot,vect_dic)
            
            # post process hip and shoulder joints
            labels[u-1,:,:,:] = aal_bp_post_proc_utils.post_process_hip_shoulders(y,img_mask,u)   
            
            # initiate alignment points for blazepose cropping
            mids_success,mid_labels = aal_bp_post_proc_utils.mid_point_reversion(angle_seq,path_saved,img_mask,u,labels,rot_params,crop_dic0,aal_bp_utilities.rot_angles,ref_width)
        
    # choose for each frame which type of alignment had outputs, medium or long images
    
    final_mids=np.zeros((mid_labels.shape[1:]))
    chosen_angle=np.zeros((mid_labels.shape[1],1))
    
    for t in range(0,mid_labels.shape[1]):
        # choose long alignment
        if sum(mid_labels[1,t,8,0:2])*sum(mid_labels[1,t,9,0:2])>0:
            final_mids[t]=mid_labels[1,t,:,:]
            chosen_angle[t,0]=angle_seq[t,1]
    
        # choose medium alignment
        elif sum(mid_labels[0,t,8,0:2])*sum(mid_labels[0,t,9,0:2])>0:
            final_mids[t]=mid_labels[0,t,:,:]
            chosen_angle[t,0]=angle_seq[t,0]
            
    print("Saving results ...")

    
        # if the user requests to save the results then save the video from midpoints 
    out_ann , path_saved = aal_bp_utilities.make_outvideo_and_path('body_auto_out_annotated.mp4','vid_test/temp/main/main_face/',main_path,frameSize)
    if saving_aux:

        aal_bp_utilities.show_midpoints_results(path_saved,out_ann,final_mids,chosen_angle)
    
    # the final stage of blazepose pipeline is to implement trained network of Blazepose on the aligned image after finding midpoints
    # initialize rotation parameters of BP as rot_params_bp and load images into data
    rot_params_bp={}
    
    # align and crop images based on midpoints extraction, if saving is true, it also saves images after aligment with mid points
    rot_params_bp,orig_jpeg_files,main_face_bp = aal_bp_post_proc_utils.align_images_midpoints(path_saved,final_mids,main_path,saving,rot_params,1,main_face)
    
    # initialize and read data 
    path_saved_bp= main_path+'vid_test/temp/main/main_face_bp/'
    data , pre_processed_labels , missing_imgs , b = aal_bp_utilities.read_data_bpoze(path_saved_bp,orig_jpeg_files,angle_seq,bpose_utils.model_body,main_face_bp)
    
    # filter and post process coordinates after blazepose
    ex_labels = aal_bp_post_proc_utils.process_bp_coordinates(pre_processed_labels)
    
    
    # if saving is on, we show the output of the blazepose results by reversing back the coordinates into the original coordinates
    out_ann , path_saved = aal_bp_utilities.make_outvideo_and_path('full_body_auto_out_annotated.mp4','vid_test/temp/main/main_face/',main_path,frameSize)
    over_all = aal_bp_post_proc_utils.reverse_back_show_coord_after_bp(path_saved,ex_labels,rot_params_bp,saving,vect_dic,final_mids,out_ann,im_size0,im_size1,main_face)
    
    print("time:",time.time()-aa)

    # track missing/out of ROI frames and substitute them
    out_ann , path_saved = aal_bp_utilities.make_outvideo_and_path('full_body_auto_out_annotated_tracked.mp4','vid_test/temp/main/main_face/',main_path,frameSize)
    over_all_coordinates_with_tracking = aal_bp_utilities.track_coordinates(path_saved,main_path,over_all,ref_width,missing_imgs,out_ann,saving,main_face)
    
    # smooth all coordinates and save the results 
    out_ann , path_saved = aal_bp_utilities.make_outvideo_and_path('full_body_auto_out_annotated_smoothed.mp4','vid_test/temp/main/main_face/',main_path,frameSize)
    over_all_coordinates_after_tracking = aal_bp_post_proc_utils.smooth_over_all_coordinates_after_tracking(over_all_coordinates_with_tracking,out_ann,saving,path_saved,main_face)
    
    print("time:",time.time()-aa)
