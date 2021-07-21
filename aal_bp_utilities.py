# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:29:05 2021

@author: Mohammad Ghahramani

required prequsits for running blazepose pipeline on a file
"""


import glob
import tensorflow as tf
import os
import numpy as np
import math
import cv2
from scipy import ndimage
import time
import matplotlib.pyplot as plt
from PIL import Image

# abbreviations and var help:
# out_ann : annotated output of the video file

# setting rotations angles and their subsequent multiplication factor rot_angles to convert them to degree

rot_angles={}
rot_angles['150']=22
rot_angles['135']=21
rot_angles['120']=20
rot_angles['105']=19
rot_angles['75']=19
rot_angles['60']=20
rot_angles['45']=21
rot_angles['30']=22

# possible angles for body alignment
angles=[0,180,90,135,150,120,105,75,60,45,30]

# set cuda usage to cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# create all folders for saving results and intermediate results
def create_all_folders(main_path):
    try:
        os.mkdir(main_path+'vid_test/')
    except:
        pass
    try:
        os.mkdir(main_path+'vid_test/temp/')
    except:
        pass
    try:
        os.mkdir(main_path+'vid_test/temp/main/')
    except:
        pass
    for ii in range(1,3):
        try:
            os.mkdir(main_path+'vid_test/temp/main/face_out_'+str(ii))
        except:
            pass
    my_list = ['face_out0','main_body','main_face','main_face_bp','main_face_II','main_final_mids','main_shoulder']
    
    for ii in my_list:
        try:
            os.mkdir(main_path+'vid_test/temp/main/'+ii)
        except:
            pass
# closing and releasing videos after writing
def close_video(saving,out_ann):
    if saving==False:
        out_ann.release()

# initiate the video object for writing them into disksbased on the given name, path, main path and framesize
def make_outvideo_and_path(video_name_ann,added_path,main_path,frameSize):

    out_ann = cv2.VideoWriter(main_path+'vid_test/temp/main/'+video_name_ann,cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize)
    path_saved = main_path + added_path

    return out_ann , path_saved

# delete intermediate results either video and images for initiating processing the new file
def deleteing_prev_results(main_path):
    
    # This function removes all previous files that were saved for detailed analysis
    files = glob.glob(main_path+'/vid_test/temp/main/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
    files = glob.glob(main_path+'/vid_test/temp/main/main_face_bp/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
    files = glob.glob(main_path+'/vid_test/temp/main/main_face/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
    files = glob.glob(main_path+'/vid_test/temp/main/main_face_II/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
       
    for ir in range(0,5):
        files = glob.glob(main_path+'/vid_test/temp/main/face_out_'+str(ir)+'/*.jpeg')
        try:
            for f in files:
                os.remove(f)
        except:
            pass
        
    files = glob.glob(main_path+'/vid_test/temp/main/face_out90/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
    files = glob.glob(main_path+'/vid_test/temp/main/main_shoulder/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
    files = glob.glob(main_path+'/vid_test/temp/main/main_final_mids/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
    files = glob.glob(main_path+'/vid_test/temp/main/face_out0/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
        
            
    files = glob.glob(main_path+'/vid_test/temp/main/face_out90l/*.jpeg')
    try:
        for f in files:
            os.remove(f)
    except:
        pass
    
    try:    
        os.remove(main_path+'/vid_test/temp/rot_info.npy')
    except:
        pass
    print('Deleted previous reports and results')

# add main path to other paths
def add_main_path(main_path,desired_path):
    return main_path+desired_path

# amend angle extreams by substituting zero and 180 due to cosine and sine
def fix_angle_extreams(u):
    if u==0: 
        return 30
    elif u==180:
        return 150
    else:
        return u

# main part of aligning the image, based on given labels and midpoints
# the rot_info dictionary contains all rotational values and they are abbreviated as:
# stpoint = starting point of cropping area
# mid_sh = mid_shoulder
# mid_hip = mid hip pointa coordinates



def align_im(img,labels):
    
    rot_info={}
    if labels.shape[1]>2.5:
        labels=labels[:,0:2]
    scale=1.1
    s_max=int(scale*max(img.shape))
    if s_max%2==1:
        s_max=s_max+1
    filler=np.zeros((s_max,s_max,3)).astype(np.uint8)
    
    # omit the same distance from mid hips side to side and as upwards

    
    # translation
    
    mid_sh = labels[0,0:2] #np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip = labels[1,0:2] #np.array([0.5*(labels[11,0]+labels[8,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
    stpoint=np.array([int(s_max/2-mid_hip[1]),int(s_max/2-mid_hip[0])])
    # filler[max(0,stpoint[0]):stpoint[0]+img.shape[0],max(stpoint[1],0):stpoint[1]+img.shape[1],:]=img[-min(0,-stpoint[0]):img.shape[0],-min(0,-stpoint[1]):img.shape[1],:]
    if stpoint[0]>0:
        a = stpoint[0]
        b = stpoint[0]+img.shape[0]
        aa=0
        bb=img.shape[0]
    else:
        a = 0
        b = img.shape[0]+[0]
        aa = -stpoint[0]
        bb = img.shape[0]
    if stpoint[1]>0:
        c = stpoint[1]
        d = stpoint[1]+img.shape[1]
        cc=0
        dd=img.shape[1]
    else:
        c = 0
        d = img.shape[1]+stpoint[1]
        cc = -stpoint[1]
        dd = img.shape[1]
    
    dd = min(img.shape[1],dd)
    d = min(filler.shape[1],d)
    
    align1= ((a-b)-(aa-bb))
    if abs(align1)>0.5 and abs(align1)<90:
        b=a+(aa-bb)
        
    align1= ((c-d)-(cc-dd))
    if abs(align1)>0.5 and abs(align1)<90:
        c=d+(cc-dd)
    
        
        
    filler[a:b,c:d,:]=img[aa:bb,cc:dd,:]

    # for u in range(labels.shape[0]):
    labels[:,0]=labels[:,0]+stpoint[1]
    labels[:,1]=labels[:,1]+stpoint[0]
    # labels[:,0] += stpoint[1]
    # labels[:,1] += stpoint[0]
    
    rot_info['add10']=-stpoint[1]
    rot_info['add11']=-stpoint[0]

    mid_sh = labels[0,0:2] #np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip = labels[1,0:2] #np.array([0.5*(labels[11,0]+labels[8,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
    body_vec = mid_hip-mid_sh

    body_vec[1]=-body_vec[1]
    body_vec=-body_vec
    
    angle=np.arcsin(body_vec[0]/(body_vec[0] ** 2+body_vec[1]**2)**0.5)
    angle_deg=math.degrees(angle)
    
    filler_rot = ndimage.rotate(filler, angle_deg,reshape=False,order=0)
    rot_info['rot_im_size']=filler_rot.shape
    rot_info['im_size']=filler.shape
    rot_info['angle']=angle

    mid_hip_old=mid_hip
    for u in range(labels.shape[0]):
        labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
    
    mid_sh = (labels[0,0:2].astype(int)) #np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip = (labels[1,0:2].astype(int)) #np.array([0.5*(labels[11,0]+labels[8,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
    
    diam=int(np.linalg.norm(mid_hip-mid_sh))
    final=filler_rot[mid_hip[0]-int(diam*2.2):mid_hip[0]+int(diam*2.2),mid_hip[1]-int(diam*1.5):mid_hip[1]+int(diam*1.7),:]
    

    labels[:,0]=labels[:,0]-(mid_hip[1]-int(diam*1.5))
    labels[:,1]=labels[:,1]-(mid_hip[0]-int(diam*2.2))
        
    rot_info['add20']=(mid_hip[1]-int(diam*1.5))
    rot_info['add21']=(mid_hip[0]-int(diam*2.2))
    

    return final,labels,rot_info

def track_partial(method,im1,im2,prev_points,new_points,to_gray,params):
    
    zeroer2 = sum(np.transpose(0.5*(1+np.sign(new_points[:,0:2]))*new_points[:,0:2]))
    zeroer1 = sum(np.transpose(0.5*(1+np.sign(prev_points[:,0:2]))*prev_points[:,0:2]))
    
    # index of points that did not exist in previous points and not to be tracked
    zeroer = (np.sign(zeroer1))
    zeroer = (np.sign(np.sign(-zeroer2+1)/2)+1)/2
    
    new_index_prev = np.multiply(np.transpose(np.tile(zeroer, (new_points.shape[1],1))), prev_points )
    # new_index # repeated
    zeroer = (np.sign(zeroer2))

    new_index_next = np.multiply(np.transpose(np.tile(zeroer, (new_points.shape[1],1))), prev_points )

    
    height, width, layers = im1.shape
    
    # hsv = np.zeros_like(im1)
    
    # hsv[..., 1] = 255
    
    frame_copy = np.copy(im2)
    # frame_copy  # repeated
    
    if to_gray:
        old_frame = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        # old_frame  # repeated
    
    flow = method(old_frame, new_frame, None)
    
    points_ii = new_index_prev  
    new_points_tracked = points_ii
    flow_1=flow.shape[1]-1
    flow_0=flow.shape[0]-1
    
    for k in range(0,points_ii.shape[0]):
        if zeroer1[k]>0 and zeroer2[k]==0: # sum(points_ii[k,:])>10 and non_zeros<(points.shape[1]/4):
            # print(ii)
            # print(flow.shape)
            # print(int(points_ii[k,1].astype(int)))
            # print(int(points_ii[k,0].astype(int)))
            new_points_tracked[k,0]=points_ii[k,0]+int(flow[min(flow_0,points_ii[k,1].astype(int)),min(points_ii[k,0].astype(int),flow_1), 0])
            new_points_tracked[k,1]=points_ii[k,1]+int(flow[min(flow_0,points_ii[k,1].astype(int)),min(points_ii[k,0].astype(int),flow_1), 1])
        if zeroer1[k]>0 and zeroer2[k]>0:
            new_points_tracked[k,0:2] = new_index_next[k,0:2]

    
    
    
    return new_points_tracked


# tracking missing frames after blazepose depends on the prvious frame
def track_partial(method,im1,im2,prev_points,new_points,to_gray,params):
    
    # find number of zero coordinates for the new and previous coordinate stored in zeroer2 and zeroer1 
    zeroer2 = sum(np.transpose(0.5*(1+np.sign(new_points[:,0:2]))*new_points[:,0:2]))
    zeroer1 = sum(np.transpose(0.5*(1+np.sign(prev_points[:,0:2]))*prev_points[:,0:2]))
    
    # index of points that did not exist in previous points and not to be tracked
    zeroer = (np.sign(zeroer1))
    zeroer = (np.sign(np.sign(-zeroer2+1)/2)+1)/2
    
    # update coordinates based on the zero coorinates
    new_index_prev = np.multiply(np.transpose(np.tile(zeroer, (new_points.shape[1],1))), prev_points )
    # new_index # repeated
    zeroer = (np.sign(zeroer2))
    
    # update coordinates based on the zero coorinates
    new_index_next = np.multiply(np.transpose(np.tile(zeroer, (new_points.shape[1],1))), prev_points )

    # store image parameters in these variables
    height, width, layers = im1.shape
    
    # copy the image for tracking
    frame_copy = np.copy(im2)
    
    # convert images to grey for tracking as it is usually required 
    if to_gray:
        old_frame = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        # old_frame  # repeated
    
    # apply tracking method to get movement directions between images
    flow = method(old_frame, new_frame, None)
    
    # initiate points coordinates for tracking
    points_ii = new_index_prev  
    new_points_tracked = points_ii
    flow_1=flow.shape[1]-1
    flow_0=flow.shape[0]-1
    
    # for all points obtain their new coordinates based on the movement calculated in flow array
    for k in range(0,points_ii.shape[0]):
        if zeroer1[k]>0 and zeroer2[k]==0: 

            new_points_tracked[k,0]=points_ii[k,0]+int(flow[min(flow_0,points_ii[k,1].astype(int)),min(points_ii[k,0].astype(int),flow_1), 0])
            new_points_tracked[k,1]=points_ii[k,1]+int(flow[min(flow_0,points_ii[k,1].astype(int)),min(points_ii[k,0].astype(int),flow_1), 1])
        if zeroer1[k]>0 and zeroer2[k]>0:
            new_points_tracked[k,0:2] = new_index_next[k,0:2]

    
    
    # return new coordinates
    return new_points_tracked

# Caclualte IOU
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# reverting back coordinates (labels) based on the rot_info dictionary
def revert_back(labels,rot_info):
    
    # first multiply coordinates based on the cropped image compared to 256 images fed to blazepose
    labels=labels[:,0:2]
    img_shape=rot_info['final_size']
    labels[:, 0] *= (img_shape[1] / 256)
    labels[:, 1] *= (img_shape[0] / 256)

    # extract the first addition parameters and add them to coordinates
    add20=rot_info['add20']
    add21=rot_info['add21']

    # for u in range(labels.shape[0]): 
    labels[:,0]=labels[:,0]+add20
    labels[:,1]=labels[:,1]+add21
       
    # extract filter rotation shape (filter_rot_sh) , imagesize and angle
    filler_rot_sh=rot_info['rot_im_size']
    filler_sh=rot_info['im_size']
    angle=rot_info['angle']
    
    # rotate back labels based on the aligned angle
    for u in range(labels.shape[0]):
        labels[u,:]=rot_back(filler_rot_sh,filler_sh,labels[u,:],-angle)


    # add cropped positions to the coordinates to bring them back to the coordinates of the original image
    stpoint1=rot_info['add10']
    stpoint0=rot_info['add11']    
    
    # for u in range(labels.shape[0]):
    labels[:,0]=labels[:,0]+stpoint1
    labels[:,1]=labels[:,1]+stpoint0
    
    # multiply coordinates based on the scale they were resized
    labels *= 1/rot_info['rot_vect'][0]

    return labels




# rotating back points considering image size before and after rotation
def rot_back(im_rot_sh,image_sh, xy, a):
    
    # finding image centers
    org_center = (np.array(image_sh[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot_sh[:2][::-1])-1)/2.
    
    # distant calc between centers
    org = xy-org_center
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center    

    
# rotating points considering image size before and after rotation

def rot(im_rot,image, xy, a):
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center   

# perform operation on labels based on the given parameters:
    # ostx : starting points of x
    # osty : starting points of y
    # osize1 : size of the first cropped image after finding the rotated body in x direction
    # osize2 : size of the first cropped image after finding the rotated body in y direction
    # oangle : calculated angle based on the chin and mid shoulders
    # oim3 is the cropped for finding the midhip
    # osecpoints : second points of the cropping area
    # ocropped_imsize0 : crooped size of the image after the second cropping in x direction
    # ocropped_imsize1 : crooped size of the image after the second cropping in y direction

    
def operate_on_labels(temp_label,ostx,osty,osize1,osize2,oangle,oim3,osec_points,ocropped_imsize0,ocropped_imsize1):
    
    # relocating labels after the first cropping
    temp_label_after = (temp_label)
    temp_label_after[:,1] -= osec_points[1]
    temp_label_after[:,0] -= osec_points[0]
    temp_label_after[temp_label_after[:,0]<0]=0
    temp_label_after[temp_label_after[:,1]<0]=0

    # rotating back based on the found angle
    for u in range(temp_label.shape[0]):
        if sum(temp_label[u,:])>0:
            temp_label[u,:]=rot_back(np.array(oim3),[osize1,osize2],temp_label[u,:],-oangle)  
    
    # relocating labels after the second cropping
    temp_label[:,1] -= osty
    temp_label[:,0] -= ostx
    temp_label[temp_label[:,0]<0]=0
    temp_label[temp_label[:,1]<0]=0
    
    return temp_label_after

# apply blazeface so that it finds small images in the upper half of the image
def apply_blazeface(vid_len,vidcap,last_frame,image,success,net,out_ann,save_path,saving):
    
    # initiating vect_dict as the dictionary contaiing face location along frames
    vect_dic={}
    main_face={}
    main_face_II={}
    main_face0 = {}
    # initiating crop_dict as the dictionary contaiing cropping area around the face for 

    crop_dic0={}
    crop_dic_dir={}
    
    # two counters c: for counting found faces, cc for counting missing faces
    c=0
    cc=0
    
    # stroing number of missed faces along the video frames
    missed_cases=0
    
    while success and c<(min(vid_len-2,last_frame)) and missed_cases<last_frame:
        if c<min(vid_len-2,last_frame):
            cc=cc+1
            
            # read and convert the image to the color space for blazeface
            I = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # prepare the image for blazefae
            I_resized=cv2.resize(I,(128,128))
            detections = net.predict_on_image(I_resized)
            vect=detections.cpu().numpy()
            
            # as the face might be small for blazeface, we need to divide the image into smaller subimages 
            # subimage 1 and its processing
            par = 2
            I1h=I[int(2*I.shape[0]/10):int(3*I.shape[0]/5),int(2.5*I.shape[1]/12):int(6*I.shape[1]/12),:]
            I_resized=cv2.resize(I1h,(128,128))
            detections = net.predict_on_image(I_resized)
            vect1=detections.cpu().numpy()
            
            # substitte the main face location if a face is found in this region and reverting the results as the original image
            if vect1.shape[0]>0:
                vect=vect1
                vect[0,range(4,16,2)] *= I1h.shape[1]/I.shape[1]
                vect[0,range(0,3,2)] *= I1h.shape[1]/I.shape[1]
                vect[0,range(5,16,2)] *= I1h.shape[0]/I.shape[0]
                vect[0,range(1,4,2)] *= I1h.shape[1]/I.shape[1]
    
                vect[0,range(4,16,2)]+=(2.5/12)
                vect[0,range(1,4,2)]+=(2.5/12)
                vect[0,range(5,16,2)]+=(par/10)
                vect[0,range(0,3,2)]+=(par/10)
    
            
            # as the face might be small for blazeface, we need to divide the image into smaller subimages 
            # subimage 2 and its processing
            
            I1h=I[int(par*I.shape[0]/10):int(3*I.shape[0]/5),int(7*I.shape[1]/12):int(10.5*I.shape[1]/12),:]
            I_resized=cv2.resize(I1h,(128,128))
            detections = net.predict_on_image(I_resized)
            vect3=detections.cpu().numpy()
            
            # substitte the main face location if a face is found in this region and reverting the results as the original image

            if vect3.shape[0]>0:
                vect=vect3
                vect[0,range(4,16,2)] *= I1h.shape[1]/I.shape[1]
                vect[0,range(0,3,2)] *= I1h.shape[1]/I.shape[1]
                vect[0,range(5,16,2)] *= I1h.shape[0]/I.shape[0]
                vect[0,range(1,4,2)] *= I1h.shape[1]/I.shape[1]
    
                vect[0,range(4,16,2)]+=(7/12)
                vect[0,range(1,4,2)]+=(7/12)
                vect[0,range(5,16,2)]+=(par/10)
                vect[0,range(0,3,2)]+=(par/10)
                
            
            # as the face might be small for blazeface, we need to divide the image into smaller subimages 
            # subimage 3 and its processing
            par = 2 
            I1h=I[int(par*I.shape[0]/10):int(3*I.shape[0]/5),int(4.75*I.shape[1]/12):int(8*I.shape[1]/12),:]
            I_resized=cv2.resize(I1h,(128,128))
            detections = net.predict_on_image(I_resized)
            vect2=np.copy(detections.cpu().numpy())
            
            # substitte the main face location if a face is found in this region and reverting the results as the original image
            if vect2.shape[0]>0:
                vect=vect2
                vect[0,range(4,16,2)] *= I1h.shape[1]/I.shape[1]
                vect[0,range(0,3,2)] *= I1h.shape[1]/I.shape[0]
                vect[0,range(5,16,2)] *= I1h.shape[0]/I.shape[0]
                vect[0,range(1,4,2)] *= I1h.shape[1]/I.shape[1]
    
                vect[0,range(4,16,2)]+=(4.75/12)
                vect[0,range(1,4,2)]+=(4.75/12)
                vect[0,range(5,16,2)]+=(par/10)
                vect[0,range(0,3,2)]+=(par/10)
    
            
            # deciding if the face exists 
            if vect.shape[0]>0:
                
                min_x=min(vect[0,range(4,16,2)])
                max_x=max(vect[0,range(4,16,2)])
                min_y=min(vect[0,range(5,16,2)])
                max_y=max(vect[0,range(5,16,2)])
                face_Ratio = (max_x-min_x)/(max_y-min_y)
                
                temp_vect=np.copy(vect[0,4:-1].reshape((6,2)))
                temp_vect[:,0] *= image.shape[1]
                temp_vect[:,1] *= image.shape[0]
                
                # if saving:
                #     # showing all information on images and video for further processing
                #     sam_face=cv2.putText(image,"FDet, #F: "+str(c).zfill(4),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                #     sam_face=cv2.putText(image,"Missed Faces, #No: "+str(missed_cases).zfill(2),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                #     sam_face=cv2.putText(image,"Face_ratio : "+str(face_Ratio),(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                    
                #     sam_face=cv2.putText(image,"Face width : "+str(max_x-min_x),(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                #     sam_face=cv2.putText(image,"Face_len : "+str((max_y-min_y)),(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
    
    
                
                    # skeleton=temp_vect.astype(int)
                    # for ii in range(skeleton.shape[0]):
                    #     if c%100>4:
                    #         cv2.circle(sam_face, center=tuple(skeleton[ii][0:2]), radius=5, color=(0, 0, 255), thickness=5)
                    #     else:
                    #         sam_face=cv2.putText(sam_face,str(ii),tuple(skeleton[ii][0:2]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                
                # face bounding box and resizing that by image size as it was normalized to 0 and 1
                
                box1=np.copy(vect[0,0:4]).reshape((2,2))
                box1[:,0] *= I.shape[0]
                box1[:,1] *= I.shape[1]
    
                # if saving:
                    # sam_face = cv2.rectangle(sam_face, tuple((int(box1[0,1]),int(box1[0,0]))),tuple((int(box1[1,1]),int(box1[1,0]))),  (200, 0, 0), 3)
                    # out_ann.write(sam_face)
                    
                # vect=detections.cpu().numpy()
                # extract face parameters for shoulder cropping 
                box1=np.copy(vect[0,0:4]).reshape((2,2))
                width=abs(box1[0,0]-box1[1,0])
                length=abs(box1[0,1]-box1[1,1])
                box1[0,0] -= width/8 
                if box1[0,0]<0:
                    box1[0,0]=0
                box1[0,1] += width/8
                box1[1,0] -= length/6
                box1[1,1] += length/3
                if box1[1,0]<0:
                    box1[1,0]=0
                box1[:,0] *= I.shape[0]
                box1[:,1] *= I.shape[1]
        
                width=abs(box1[0,0]-box1[1,0])
                length=abs(box1[0,1]-box1[1,1])
                ratio1=25/length
                box1 *= ratio1
                width *= ratio1
                length *= ratio1
        
                # a temporary vector is accumulating parameters to be saved in vect dictionary for later usage
                temp_vect=np.copy(vect[0,4:-1].reshape((6,2)))
                temp_vect[:,0] *= image.shape[1]
                temp_vect[:,1] *= image.shape[0]
                
                # the image is resized so that it forces less compuation for rotation, cropping and ....
                II=cv2.resize(I,(int(I.shape[1]*ratio1),int(I.shape[0]*ratio1)))
                
                # stx0 is the starting point of x direction 
                stx0=int(max(0,box1[0,0]-1.5*width))
                
                # stx0 is the starting point of x direction 
                sty0=int(max(0,box1[0,1]-2*length))
                
                # crop the part of the image where the person is 
                sam=II[stx0:int(min(II.shape[0],box1[0,0]+2.5*width)),
                      sty0:int(min(II.shape[1],box1[0,1]+3*length)),:]
                
                # store image parameters and face location and coordinates, parameters to store them in cropdic0 as
                # the crop dictionary at zero rotation and crop_dic_dir storing face box bonding box information
                crop_dic0[str(c)]=[ratio1,stx0,sty0,int(sam.shape[0]),int(sam.shape[1])]
                crop_dic_dir[str(c)]=[box1.reshape(1,4),width,length]

                # cv2.imwrite(save_path+'vid_test/temp/main/face_out0/face_out_'+str(c).zfill(4)+'.jpeg',sam)
                main_face0['face_out_'+str(c).zfill(4)+'.jpeg']= sam
                if True:
                # out.write(I)
                    main_face['face_out_'+str(c).zfill(4)+'.jpeg']= I #cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(save_path+'vid_test/temp/main/main_face/face_out_'+str(c).zfill(4)+'.jpeg',I)
                    main_face_II['face_out_'+str(c).zfill(4)+'.jpeg'] = II #cv2.cvtColor(II, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(save_path+'vid_test/temp/main/main_face_II/face_out_'+str(c).zfill(4)+'.jpeg',II)
                    
                # storing vect dictionary of vector 
                vect_dic[str(c)]=vect
                # cc=cc+1
                c=c+1
                # if c%50==0:
                    # print('found #{} face '.format(c))
                    # print('Lost #{} face '.format(missed_cases))
    
            else:
                # report number of fond and missing faces
                missed_cases += 1
                # cc=cc+1
                # print("missed frame number:", cc)
                # if missed_cases%50==0:
                #     print('Lost #{} face '.format(missed_cases))
                #     print('found #{} face '.format(c))
            success,image = vidcap.read()
    
    try:
        out_ann.release()
    except:
        pass
    return (vect_dic,     crop_dic0,    crop_dic_dir,main_face,main_face_II,main_face0)

# read and load images after finding midpoints in the first or second phase
def read_images_after_blazeface(path_saved,vect_dic,main_faces):
    # jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    jpegfiles = sorted(main_faces.keys())
    data=np.zeros((len(jpegfiles),256,256,3))
    
    # read images and cnovert them to tensors 
    for t in range(len(jpegfiles)):
        vect=vect_dic[str(t)]
        if  vect.shape[0]>0:

            # #Add dims to rgb_tensor
            # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
            # img1 = np.array(cv2.imread(path_saved+jpegfiles[t],-1))
            img1 = main_faces[jpegfiles[t]]
            img_ten = tf.convert_to_tensor(img1, dtype=tf.float32)
            img_ten1 = tf.expand_dims(img_ten , 0)
            # img = tf.io.read_file(path_saved+jpegfiles[t])
            # img = tf.image.decode_image(img_ten1)
            data[t,...] = tf.image.resize(img_ten1, [256, 256])
    
    return data

# filter keypoints after extraction
def post_process_keypoints(y):
    ex_labels=np.zeros((y.shape[0],y.shape[3],3))
    for p in range(y.shape[0]):   
        for q in range(y.shape[3]):
            try_var=(y[p,:,:,q])
            ind = np.unravel_index(np.argmax(try_var, axis=None), try_var.shape)  # returns a tuple
            ex_labels[p,q,0:2]=np.array(list([ind[1],ind[0]]))*2
            if sum(ex_labels[p,q,0:2])<30:
                ex_labels[p,q,0:2]=0
                
    return ex_labels

# show keypoints on the original image based on the extracted labels, cropping parameters dictionary path to save the video

def show_shoulder_results(ex_labels,crop_dic0,out_ann,video_name_ann,path_saved,path_saved_shoulder,main_path):
    
    jpegfiles = list()
    jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    
    
    for t in range(len(jpegfiles)-1):
        I1=cv2.imread(path_saved+jpegfiles[t])
    
        skeleton=np.copy(ex_labels[t])
        zeroer=sum(np.transpose(skeleton))<10
        skeleton[:,0]/=256/(crop_dic0[str(t)][4])
        skeleton[:,1]/=256/(crop_dic0[str(t)][3])
        
        skeleton[:,0]+=crop_dic0[str(t)][2]
        skeleton[:,1]+=crop_dic0[str(t)][1]
        skeleton *= 1/crop_dic0[str(t)][0]           
    
        samasl=np.copy(I1) 
        skeleton=skeleton.astype(int)
        skeleton[zeroer,:]=0
        for ii in range(skeleton.shape[0]):
            if sum(skeleton[ii][0:2])>5:
                if t%100!=0:
                    cv2.circle(samasl, center=tuple(skeleton[ii][0:2]), radius=5, color=(0, 0, 255), thickness=3)
                else:
                    cv2.putText(samasl,str(ii),tuple(skeleton[ii][0:2]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 )
    
                # cv2.putText(samasl,str(ii),tuple(skeleton[ii][0:2]),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2)
        samasl=cv2.putText(samasl,"Shoulder_net, #F: "+str(t).zfill(4),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
    
        # cv2.imwrite(main_path+'vid_test/temp/main/main_shoulder/shoulder_out_'+str(t).zfill(4)+'.jpeg',cv2.cvtColor(samasl, cv2.COLOR_BGR2RGB))
    
        out_ann.write(cv2.cvtColor(samasl, cv2.COLOR_BGR2RGB))
    
    try:
        out_ann.release()
    except:
        pass

# show extracted midpoints before alignment in a video saved showing the extracted angle, midpoints and output annotated video
def show_midpoints_results(path_saved,out_ann,final_mids,chosen_angle):
    
    jpegfiles = list()
    jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])

    
    for t in range(len(jpegfiles)):
        I1=cv2.imread(path_saved+jpegfiles[t])
        skeleton=np.copy(final_mids[t])
        zeroer=sum(np.transpose(skeleton))<10

        samasl=np.copy(I1) 
        skeleton=skeleton.astype(int)
        skeleton[zeroer,:]=0
        samasl=cv2.putText(samasl,"Directional_net, #F: "+str(t).zfill(4)+'_'+str(chosen_angle[t,0]),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
        for ii in range(skeleton.shape[0]):
            if sum(skeleton[ii][0:2])>5:
                cv2.circle(samasl, center=tuple(skeleton[ii][0:2]), radius=5, color=(0, 0, 255), thickness=3)
                
        out_ann.write(cv2.cvtColor(samasl, cv2.COLOR_BGR2RGB))
    
    try:
        out_ann.release()
    except:
        pass

# Filter extracted keypoints, 
# clean the data
# substitute midpoints if one points is missing
# as there are multiple mid points they are all calculated and substituted if one of the 3 points is missing
# 

def calculate_body_angle(ex_labels):
    angle_seq=np.zeros((ex_labels.shape[0],2))
    for t in range(ex_labels.shape[0]):
        if ex_labels[t,10,0]==0 and ex_labels[t,4,0]*ex_labels[t,6,0]>0:
            ex_labels[t,10,0:2]=(ex_labels[t,4,0:2]+ex_labels[t,6,0:2])/2
        if ex_labels[t,2,0]*ex_labels[t,3,0]>0:
            upper_mid=(ex_labels[t,2,0:2]+ex_labels[t,3,0:2])/2
        else:
            if ex_labels[t,2,0]+ex_labels[t,3,0]>0:
                upper_mid=(ex_labels[t,2,0:2]+ex_labels[t,3,0:2])
            else:
                angle_seq[t,0]=90
        
        if ex_labels[t,9,0]>0:
            down_mid= ex_labels[t,9,0:2]           
        else:
            if ex_labels[t,10,0]>0:
                down_mid= ex_labels[t,10,0:2] 
            else:
                angle_seq[t,0]=90
        if angle_seq[t,0]==0:
            down_vect=down_mid-upper_mid
            true_angle=np.angle(complex(down_vect[0],down_vect[1]), deg=True) 
            pos = (np.abs(angles-true_angle)).argmin()
            angle_seq[t,0]=angles[pos]
        
        if ex_labels[t,8,0]*ex_labels[t,9,0]>0:
            down_vect=ex_labels[t,9,0:2]-ex_labels[t,8,0:2]
            true_angle=np.angle(complex(down_vect[0],down_vect[1]), deg=True) 
            pos = (np.abs(angles-true_angle)).argmin()
            angle_seq[t,1]=angles[pos]
        else:
            if ex_labels[t,9,0]*ex_labels[t,10,0]>0:
                down_vect=ex_labels[t,9,0:2]-ex_labels[t,10,0:2]
                true_angle=np.angle(complex(down_vect[0],down_vect[1]), deg=True) 
                pos = (np.abs(angles-true_angle)).argmin()
                angle_seq[t,1]=angles[pos]
            else:
                if ex_labels[t,8,0]*ex_labels[t,10,0]>0:
                    down_vect=ex_labels[t,10,0:2]-ex_labels[t,8,0:2]
                    true_angle=np.angle(complex(down_vect[0],down_vect[1]), deg=True) 
                    pos = (np.abs(angles-true_angle)).argmin()
                    angle_seq[t,1]=angles[pos]
                else:
                    angle_seq[t,1]=90
    return angle_seq

#  read and load image data into tensors for the main blazepose

def read_data_bpoze(path_saved,orig_jpegfiles,angle_seq,model_body,main_face):
    
    jpegfiles = list()
    # jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    jpegfiles = sorted(main_face.keys())
    y = np.zeros((len(orig_jpegfiles),256,256,13)).astype(np.uint8)

    data=np.zeros((angle_seq.shape[0],256,256,3))
    # aa = time.time()
    missing_imgs=list()
    for t in range(len(orig_jpegfiles)):
        # print(t)
        if orig_jpegfiles[t] in jpegfiles:
            
            img1 = main_face[orig_jpegfiles[t]]
            img_ten = tf.convert_to_tensor(img1, dtype=tf.float32)
            img_ten1 = tf.expand_dims(img_ten , 0)
            # img = tf.io.read_file(path_saved+orig_jpegfiles[t])
            # img = tf.image.decode_image(img)
            data[t,...] = tf.image.resize(img_ten1, [256, 256])
        else:
            # print(t)
            missing_imgs.append(t)
    b=time.time()

    # a=time.time()
    y = model_body.predict(data)
    # print('hdd read', b-aa)
    # print('time on calculating original blazepose on the wholebody',b-a)
                
    return data , y , missing_imgs,b

# The main function that tracks coordinates in frames that either the face is not detected, 
# miss alignment or the roi is distant enough that is out of human body abilities from frames to frames

def track_coordinates(path_saved,main_path,over_all,ref_width,missing_imgs,out_ann,saving,main_faces):

    jpegfiles = list()
    jpegfiles = sorted(main_faces.keys())
    # jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])

    
    
    # initiate over_all coordinates going to be smoothed or tracked after blazepose 
    over_all_smoothed=np.copy(over_all)
    over_all_smoothed[over_all_smoothed>ref_width]=ref_width-1
    
    # read images
    for t in range(len(jpegfiles)):
        img = main_faces[jpegfiles[t]] # cv2.imread(path_saved+jpegfiles[t])
        # orig_img = np.copy(img)
        if t==0:
            prev_image = (img)
            # orig_prev_img = np.copy(orig_img)
    
            # had_to_be_tracked = False
        
        # initialize previous frames and coordinates for tracking
        skeleton = np.copy(over_all_smoothed[t,:,:].astype(int))
        prev_skeleton = np.zeros((skeleton.shape))
        prev_selected = False
        
        if t>0:
            # if the first frame is passed then consider checking for tracking
            if sum(over_all_smoothed[t-1,:,0].astype(int)/10)>(skeleton.shape[0]*10):
                # if not had_to_be_tracked:
                prev_skeleton = np.copy(over_all_smoothed[t-1,:,:].astype(int))
                prev_selected = True
    
        # if saving:
        #     img=cv2.putText(img,"Full_Body_smoother, #F: "+str(t).zfill(4),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
        
        # compare zero indices from previous frame and current frame 
        zeroer2 = np.sign(sum(np.transpose(0.5*(1+np.sign(skeleton[:,0:2]))*skeleton[:,0:2]))) 
        zeroer1 =  (np.sign(sum(np.transpose(0.5*(1+np.sign(prev_skeleton[:,0:2]))*prev_skeleton[:,0:2]))))
        zeroer22 = np.copy(zeroer2)
        zeroer11 = np.copy(zeroer1)
        
        # do not track face keypoints as they were previously by blazeface
        # zeroer22[13:19]=0
        # zeroer11[13:19]=0
        
        # initialize the method object for tracking
        method = cv2.optflow.calcOpticalFlowSparseToDense
        
        # if we have passed the first frame then calculate the euclidian distance of the coordinates from frame to frame
        if t>0 and prev_selected == True:
            jump_cond = np.linalg.norm(skeleton[:,0:2]-prev_skeleton[:,0:2])
    
            # check if no out of image size coordinates have appeared after tracking
            prev_skeleton[prev_skeleton>1279]=1279
            skeleton[skeleton>1279]=1279
            skeleton[skeleton<0]=0
            prev_skeleton[prev_skeleton<0]=0
            
            # the main part of the paper is implemented here:
            # it is totally unclear what conditions tracker switches between blazepose and tracker but the following conditions 
            # could successfully switch between them and acheive comparable performance 
            # 1. if the Euclidian distance between non-zero coordinates in between frames exceeds than a threshold
            # 2. if the new frame does not have the detected face
            # 3. if threr is a face but there are missing coordinates compared to the previous frame
            
            if jump_cond>400 and (sum(zeroer2)<sum(zeroer1)-5 or (t-1) in missing_imgs) or sum(zeroer2)<7: # or abs(pixel_means-pixel_means_prev)>40):  #or (IOU<0.85)
                # had_to_be_tracked = True 
                # print(t)
                
                # min_prev_x = int(min(prev_skeleton[:,0])*0.8)
                    
                # if prev_image.shape[1]>prev_image.shape[0]:
                #     cropped_prev_img = prev_image[:,min_prev_x:min_prev_x+int(prev_image.shape[1]/3),:]
                #     cropped_img = img[:,min_prev_x:min_prev_x+int(prev_image.shape[1]/3),:]
                #     cropped_prev_img = cv2.resize(cropped_prev_img,tuple((int(cropped_prev_img.shape[1]/2),int(cropped_prev_img.shape[0]/2))),interpolation=cv2.INTER_NEAREST)
                #     cropped_img = cv2.resize(cropped_img,tuple((int(cropped_img.shape[1]/2),int(cropped_img.shape[0]/2))),interpolation=cv2.INTER_NEAREST)
                    
                #     prev_skeleton[:,0] -= min_prev_x
                #     prev_skeleton[:,0:2] = 0.5 * prev_skeleton[:,0:2]
                    
                #     skeleton[:,0] -= min_prev_x
                #     skeleton[:,0:2] = 0.5 * skeleton[:,0:2]
                    
                #     prev_skeleton[:,0] -= min_prev_x
                #     prev_skeleton[:,0:2] = 0.5 * prev_skeleton[:,0:2]
    
                # else:
                #     cropped_prev_img = cv2.resize(prev_image,tuple((int(prev_image.shape[1]/3),int(prev_image.shape[0]/3))),interpolation=cv2.INTER_NEAREST)
                #     cropped_img = cv2.resize(img,tuple((prev_image.shape[1],prev_image.shape[0])),interpolation=cv2.INTER_NEAREST)
                    
                #     prev_skeleton[:,0:2] *= 0.333333
                #     skeleton[:,0:2] *= 0.333333
    
                # # aa=time.time()
                # tracked_points=track_partial(method,cropped_prev_img,cropped_img,prev_skeleton,skeleton,True,[])
                # # time.time()-aa
                
                # if prev_image.shape[1]>prev_image.shape[0]:
    
                #     tracked_points[:,0:2] = 2 * tracked_points[:,0:2]
                #     tracked_points[:,0] += min_prev_x
                #     skeleton[:,0:2] = 2 * skeleton[:,0:2]
                #     skeleton[:,0] += min_prev_x
    
    
                # else:
                #     # cropped_prev_img = cv2.resize(prev_image,tuple((prev_image.shape[1]/3).astype(int),(prev_image.shape[0]/3).astype(int)),interpolation=cv2.INTER_NEAREST)
                #     # cropped_img = cv2.resize(img,tuple(prev_image.shape[1],prev_image.shape[0]),interpolation=cv2.INTER_NEAREST)
                    
                #     tracked_points[:,0] *= 3
                    
                # aa=time.time()
                tracked_points=track_partial(method,prev_image,img,prev_skeleton,skeleton,True,[])
                # time.time()-aa
                over_all_smoothed[t,:,:]=tracked_points
                # method,im1,im2,prev_points,new_points
                


                skeleton_nontracked = np.copy(skeleton)
                skeleton_face = np.copy(skeleton[13:19,0:2])
                skeleton = np.copy(tracked_points)
                        
                 
                if sum(skeleton_face[:,0])>200:
                    skeleton[13:19,0:2] = np.copy(skeleton_face)
                
                # if saving:
                #     for ii in range(0,skeleton.shape[0]):
                #         if skeleton[ii][0]>20:
                #             cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=3, color=(0, 255, 0), thickness=3)
        
                #     img=cv2.putText(img,"Changes, #F: "+str(jump_cond),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                #     # samasl=cv2.putText(img,"Tracked , Green circles: ",(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2 )
                #     # samasl=cv2.putText(img,"Current , Blue Circles: ",(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 )
                #     # samasl=cv2.putText(img,"Previous, Red Circles: ",(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                #     out_ann.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
            # else:
                
                
            #     if saving:
            #         for ii in range(skeleton.shape[0]):      
            #             cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=3, color=(0, 255, 0), thickness=3)
        
        # if prev_selected and saving:
        #     img=cv2.putText(img,"Changes, #F: "+str(jump_cond),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
        
        # if saving:
        #     # cv2.imwrite(main_path+'vid_test/temp/main/main_body/final_body_'+str(t).zfill(4)+'.jpeg',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     out_ann.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        prev_image = img
        # orig_prev_img = orig_img
    try:
        out_ann.release()
    except:
        pass
    
    np.save("skeleton_after_tracking",skeleton)

    return over_all_smoothed

# The main function that tracks coordinates in frames that either the face is not detected, 
# miss alignment or the roi is distant enough that is out of human body abilities from frames to frames

def track_coordinates1(path_saved,main_path,over_all,ref_width,missing_imgs,out_ann,saving):

    jpegfiles = list()
    jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    
    # initiate over_all coordinates going to be smoothed or tracked after blazepose 
    
    
    over_all_smoothed=np.copy(over_all)
    over_all_smoothed[over_all_smoothed>ref_width]=ref_width-1
    
    
    for t in range(len(jpegfiles)):
        img = cv2.imread(path_saved+jpegfiles[t])
        
        orig_img = np.copy(img)
        # out_ann1.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if t==0:
            prev_image = (img)
            orig_prev_img = np.copy(orig_img)
    
            had_to_be_tracked = False
        
        skeleton = np.copy(over_all_smoothed[t,:,:].astype(int))
        prev_skeleton = np.zeros((skeleton.shape))
        prev_selected = False
        
        if t>0:
            if sum(over_all_smoothed[t-1,:,0].astype(int))>(skeleton.shape[0]*10):
                # if not had_to_be_tracked:
                prev_skeleton = over_all_smoothed[t-1,:,:].astype(int)
                # else:
                #     prev_skeleton = tracked_points
                prev_selected = True
                
        # for ii in range(13,skeleton.shape[0]):
        #     if True:
                # cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(255, 0, 0), thickness=3)
                
        # for ii in range(skeleton.shape[0]):      
        #         # cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=3, color=(0, 255, 0), thickness=3)
    
        #     cv2.putText(img,str(ii),tuple(skeleton[ii][0:2].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2 )
        #     # cv2.imwrite('c:/downloads/vid_test/temp/main/main_body/final_body_'+str(ii)+str(t).zfill(4)+'.jpeg',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        #     # skeleton[6:8,:]=0
        #     # over_all[t,19:,0:2]=skeleton[4:,0:2]
    
    
        # samasl=cv2.putText(img,"Full_Body_smoother, #F: "+str(t).zfill(4)+'_'+str(chosen_angle[t,0]),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
        
        zeroer2 = np.sign(sum(np.transpose(0.5*(1+np.sign(skeleton[:,0:2]))*skeleton[:,0:2]))) 
        zeroer1 =  (np.sign(sum(np.transpose(0.5*(1+np.sign(prev_skeleton[:,0:2]))*prev_skeleton[:,0:2]))))
        
        method = cv2.optflow.calcOpticalFlowSparseToDense
        
        if t>0 and prev_selected == True:
            jump_cond = np.linalg.norm(skeleton[:,0:2]-prev_skeleton[:,0:2])
            # boxA = [min(skeleton[zeroer2>0.5,0]),min(skeleton[zeroer2>0.5,1]),max(skeleton[zeroer2>0.5,0]),max(skeleton[zeroer2>0.5,1])]
            # boxB = [min(prev_skeleton[zeroer2>0.5,0]),min(prev_skeleton[zeroer2>0.5,1]),max(prev_skeleton[zeroer2>0.5,0]),max(prev_skeleton[zeroer2>0.5,1])]
    
            # IOU = bb_intersection_over_union(boxA, boxB)
            
            prev_skeleton[prev_skeleton>1279]=1279
            skeleton[skeleton>1279]=1279
            

            skeleton[skeleton<0]=0
            prev_skeleton[prev_skeleton<0]=0
            
            
            
            # pixel_means =  np.mean(orig_img[skeleton[zeroer2>0.5,1].astype(int)-1,skeleton[zeroer2>0.5,0].astype(int)-1,:])
            # pixel_means_prev = np.mean(orig_prev_img[prev_skeleton[zeroer1>0.5,1].astype(int)-1,prev_skeleton[zeroer1>0.5,0].astype(int)-1,:])
            
            if jump_cond>400 and sum(zeroer2)<sum(zeroer1):  #or (IOU<0.85)
                # had_to_be_tracked = True 
                # print(t)
                
                min_prev_x = int(min(prev_skeleton[:,0])*0.8)
                    
                if False: #prev_image.shape[1]>prev_image.shape[0]:
                    cropped_prev_img = prev_image[:,min_prev_x:min_prev_x+int(prev_image.shape[1]/3),:]
                    cropped_img = img[:,min_prev_x:min_prev_x+int(prev_image.shape[1]/3),:]
                    cropped_prev_img = cv2.resize(cropped_prev_img,tuple((int(cropped_prev_img.shape[1]/2),int(cropped_prev_img.shape[0]/2))),interpolation=cv2.INTER_NEAREST)
                    cropped_img = cv2.resize(cropped_img,tuple((int(cropped_img.shape[1]/2),int(cropped_img.shape[0]/2))),interpolation=cv2.INTER_NEAREST)
                    
                    prev_skeleton[:,0] -= min_prev_x
                    prev_skeleton[:,0:2] = 0.5 * prev_skeleton[:,0:2]
                    
                    skeleton[:,0] -= min_prev_x
                    skeleton[:,0:2] = 0.5 * skeleton[:,0:2]
                    
                    prev_skeleton[:,0] -= min_prev_x
                    prev_skeleton[:,0:2] = 0.5 * prev_skeleton[:,0:2]
    
                else:
                    cropped_prev_img = cv2.resize(prev_image,tuple((int(prev_image.shape[1]/3),int(prev_image.shape[0]/3))),interpolation=cv2.INTER_NEAREST)
                    cropped_img = cv2.resize(img,tuple((int(img.shape[1]/3),int(img.shape[0]/3))),interpolation=cv2.INTER_NEAREST)
                    
                    prev_skeleton[:,0:2] = 0.333333 *  prev_skeleton[:,0:2]
                    skeleton[:,0:2] = 0.333333 * skeleton[:,0:2]
                
                prev_skeleton[prev_skeleton>1279]=1279
                skeleton[skeleton>1279]=1279
                
                skeleton[skeleton<0]=0
                prev_skeleton[prev_skeleton<0]=0
                
                where_are_NaNs = np.isnan(prev_skeleton)
                prev_skeleton[where_are_NaNs] = 0
                
                where_are_NaNs = np.isnan(skeleton)
                skeleton[where_are_NaNs] = 0
    
                # aa=time.time()
                tracked_points = skeleton 
                try:
                    tracked_points=track_partial(method,cropped_prev_img,cropped_img,prev_skeleton,skeleton,True,[])
                # time.time()-aa
                except:
                    print(t)
                    pass
                
                if False: #prev_image.shape[1]>prev_image.shape[0]:
    
                    tracked_points[:,0:2] = 2 * tracked_points[:,0:2]
                    tracked_points[:,0] += min_prev_x
                    skeleton[:,0:2] = 2 * skeleton[:,0:2]
                    skeleton[:,0] += min_prev_x
    
    
                else:
                    # cropped_prev_img = cv2.resize(prev_image,tuple((prev_image.shape[1]/3).astype(int),(prev_image.shape[0]/3).astype(int)),interpolation=cv2.INTER_NEAREST)
                    # cropped_img = cv2.resize(img,tuple(prev_image.shape[1],prev_image.shape[0]),interpolation=cv2.INTER_NEAREST)
                    
                    tracked_points[:,0:2] *= 3
                    
                # aa=time.time()
                # tracked_points=track_partial(method,prev_image,img,prev_skeleton,skeleton,True,[])
                # time.time()-aa
                over_all_smoothed[t,:,:]=tracked_points
                # method,im1,im2,prev_points,new_points
                
                
                
                # im1 = prev_image
                # im2 = img
                # prev_points = prev_skeleton
                # new_points = skeleton
                
                skeleton_nontracked = np.copy(skeleton)
                skeleton_face = np.copy(skeleton[13:19,0:2])
                skeleton = np.copy(tracked_points)
                
                if sum(skeleton_face[:,0])>200:
                    skeleton[13:19,0:2] = np.copy(skeleton_face)
                
                # if saving:
                #     for ii in range(0,skeleton.shape[0]):
                #         if skeleton[ii][0]>20:
                #             cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=3, color=(0, 255, 0), thickness=3)
        
                    # img=cv2.putText(img,"Changes, #F: "+str(jump_cond),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                    # # samasl=cv2.putText(img,"Tracked , Green circles: ",(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2 )
                    # # samasl=cv2.putText(img,"Current , Blue Circles: ",(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 )
                    # # samasl=cv2.putText(img,"Previous, Red Circles: ",(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
                    # out_ann.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # else:
            #     had_to_be_tracked = False
    
            # else:
                
                
                # if saving:
                #     for ii in range(skeleton.shape[0]):      
                #         cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=3, color=(0, 255, 0), thickness=3)
        
        # if prev_selected and saving:
        #     img=cv2.putText(img,"Changes, #F: "+str(jump_cond),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
        
        # if saving:
        #     # cv2.imwrite(main_path+'vid_test/temp/main/main_body/final_body_'+str(t).zfill(4)+'.jpeg',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     out_ann.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        prev_image = img
        # orig_prev_img = orig_img
    try:
        out_ann.release()
    except:
        pass
    # np.save("skeleton_after_tracking",skeleton)

    return over_all_smoothed
    




    
    # prev_image = np.zeros_like()
    
    # os.chdir('C:/Downloads/Optical-Flow-in-OpenCV/algorithms/')
    
    
        
        
                
        # for ii in range(13,skeleton.shape[0]):
        #     if True:
                # cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(255, 0, 0), thickness=3)
                

        

                # method,im1,im2,prev_points,new_points
                
                
                
                # im1 = prev_image
                # im2 = img
                # prev_points = prev_skeleton
                # new_points = skeleton
                
     
    
    
    
# it might be useful if plotting blazeface extracted coordinates written originally by Bface
# def plot_detections(img, detections, larger=True,with_keypoints=True):
    #     fig, ax = plt.subplots(1, figsize=(8, 8))
    #     ax.grid(False)
    #     ax.imshow(img)
        
    #     if isinstance(detections, torch.Tensor):
    #         detections = detections.cpu().numpy()
    
    #     if detections.ndim == 1:
    #         detections = np.expand_dims(detections, axis=0)
    
            
    #     for i in range(detections.shape[0]):
    #         ymin = detections[i, 0] * img.shape[0]
    #         xmin = detections[i, 1] * img.shape[1]
    #         ymax = detections[i, 2] * img.shape[0]
    #         xmax = detections[i, 3] * img.shape[1]
    
    #         rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                  linewidth=1, edgecolor="r", facecolor="none", 
    #                                  alpha=detections[i, 16])
    #         ax.add_patch(rect)
    
    #         if with_keypoints:
    #             for k in range(6):
    #                 kp_x = detections[i, 4 + k*2    ] * img.shape[1]
    #                 kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
    #                 if larger==False:
    #                     circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
    #                                             edgecolor="lightskyblue", facecolor="none", 
    #                                             alpha=detections[i, 16])
    #                 else:
    #                     circle = patches.Circle((kp_x, kp_y), radius=2.5, linewidth=3, 
    #                         edgecolor="lightskyblue", facecolor="none", 
    #                         alpha=detections[i, 16])
    #                 ax.add_patch(circle)
            
    #     plt.show()
    
    
