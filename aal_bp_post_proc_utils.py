# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:31:29 2021

@author: Moha

Post_processing utils
"""

import os
import numpy as np
import cv2
import math
from scipy import ndimage
import pandas as pd
import time
import fir1
from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin
from scipy.signal import butter, lfilter, freqz


# defining butterworth filters
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# defining applying a butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


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


# amend angle extreams by substituting zero and 180 due to cosine and sine
def fix_angle_extreams(u):
    if u==0: 
        return 30
    elif u==180:
        return 150
    else:
        return u
    
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

    # filler[stpoint[0]:stpoint[0]+img.shape[0],stpoint[1]:stpoint[1]+img.shape[1],:]=img


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
    # img = cv2.line(img,tuple(mid_hip),tuple(mid_sh),(255,0,0),5)
    body_vec[1]=-body_vec[1]
    body_vec=-body_vec
    
    angle=np.arcsin(body_vec[0]/(body_vec[0] ** 2+body_vec[1]**2)**0.5)
    angle_deg=math.degrees(angle)
    
    filler_rot = ndimage.rotate(filler, angle_deg,reshape=False,order=0)
    rot_info['rot_im_size']=filler_rot.shape
    rot_info['im_size']=filler.shape
    rot_info['angle']=angle
    
    # if body_vec[0]<0:
    #     angle=angle+90
    mid_hip_old=mid_hip
    for u in range(labels.shape[0]):
        labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
    
    mid_sh = (labels[0,0:2].astype(int)) #np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip = (labels[1,0:2].astype(int)) #np.array([0.5*(labels[11,0]+labels[8,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
    
    diam=int(np.linalg.norm(mid_hip-mid_sh))
    final=filler_rot[mid_hip[0]-int(diam*2.2):mid_hip[0]+int(diam*2.2),mid_hip[1]-int(diam*1.5):mid_hip[1]+int(diam*1.7),:]
    


    # for u in range(labels.shape[0]):
        # labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
    labels[:,0]=labels[:,0]-(mid_hip[1]-int(diam*1.5))
    labels[:,1]=labels[:,1]-(mid_hip[0]-int(diam*2.2))
        
    rot_info['add20']=(mid_hip[1]-int(diam*1.5))
    rot_info['add21']=(mid_hip[0]-int(diam*2.2))
    
    # labels[:,0] += (-(mid_hip[1]-int(diam*1.5)))
    # labels[:,1] += (-(mid_hip[0]-int(diam*2.2)))

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


#storing cropping parameters for rotating the image to prepare it aligned for mid hip extraction
def mid_hip_image(sam,saml,stx,sty,size1,size2,sec_points,im3,stxl,styl,size1l,size2l,sec_pointsl,im3l):
    
    # storing the following parameters in a dictionary:
        # stx: starting point of cropping in direction x
        # sty: starting point of cropping in direction y
        # size1: image size after croping in direction x
        # size2: image size after croping in direction y
        # sec_points: secondary points created for angular croping
        # im3: intermediate image for angular croping
        # sam0: size of the intermediate image after angular croping and the angle along x axis
        # sam1: size of the intermediate image after angular croping and the angle along y axis

        
        
    temp_dict={}
    temp_dict['stx']=stx
    temp_dict['sty']=sty
    temp_dict['size1']=size1
    temp_dict['size2']=size2
    temp_dict['sec_points']=sec_points
    temp_dict['im3']=im3
    temp_dict['sam0']=sam.shape[0]
    temp_dict['sam1']=sam.shape[1]
    

    temp_dictl={}
    temp_dictl['stx']=stxl
    temp_dictl['sty']=styl
    temp_dictl['size1']=size1l
    temp_dictl['size2']=size2l
    temp_dictl['sec_points']=sec_pointsl
    try:
        temp_dictl['sam0']=saml.shape[0]
        temp_dictl['sam1']=saml.shape[1]
    except:
        temp_dictl['sam0']=0
        temp_dictl['sam1']=0
    temp_dictl['im3']=im3l
    
    return temp_dict,temp_dictl

# Saving files after finding midhip points

def save_files_for_midhips(file_name,file_namel,z,sam,saml):
    # file_name=main_path+'vid_test/temp/main/face_out_1/'+jpegfiles[t][0:-5]+'_01'+'.jpeg'
    # file_namel=main_path+'vid_test/temp/main/face_out_2/'+jpegfiles[t][0:-5]+'_02'+'.jpeg'
    image_mask=np.zeros((1,2))
    try:
        if abs(z)<90:
            sam2=cv2.rotate(sam, cv2.cv2.ROTATE_90_CLOCKWISE)
            # cv2.imwrite(file_name,sam2)
            image_mask[0,0]=1
            if z!=0 and z!=180:
                sam2=cv2.rotate(saml, cv2.cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(file_namel,sam2)
                image_mask[0,1]=1
    
        else:
            if z!=90:
                sam2=cv2.rotate(sam, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                # cv2.imwrite(file_name,sam2)
                image_mask[0,0]=1
    
                
                if z!=0 and z!=180:
    
                    sam2=cv2.rotate(saml, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(file_namel,sam2)
                image_mask[0,1]=1
    
            else:
                sam2=sam
                # cv2.imwrite(file_name,sam2)
                image_mask[0,0]=1
    
    
                sam2=saml
                cv2.imwrite(file_namel,sam2)
                image_mask[0,1]=1
    except:
        pass

    return image_mask

# filter and post process mid hip and mid shoulders in terms of substituting mid or side points if one of them is absent
def post_process_hip_shoulders(y,img_mask,u):
    
    
    ex_labels90l=np.zeros((y.shape[0],y.shape[3],3))
    for p in range(y.shape[0]):   
        if img_mask[u-1,p]>0:
            for q in range(y.shape[3]):
                try_var=(y[p,:,:,q])
                ind = np.unravel_index(np.argmax(try_var, axis=None), try_var.shape)  # returns a tuple
                ex_labels90l[p,q,0:2]=np.array(list([ind[1],ind[0]]))*2
                if q==9 and ex_labels90l[p,q,0]==0:
                    if  ex_labels90l[p,6,0]*ex_labels90l[p,7,0]>0:
                        ex_labels90l[p,q,0:2]=(ex_labels90l[p,6,0:2]+ex_labels90l[p,9,0:2])/2

                if q==10 and ex_labels90l[p,q,0]==0:
                    if  ex_labels90l[p,9,0]*ex_labels90l[p,8,0]>0:
                        ex_labels90l[p,q,0:2]=(ex_labels90l[p,9,0:2]+ex_labels90l[p,8,0:2])/2
                else: 
                    if q==10 and ex_labels90l[p,9,0]==0 and ex_labels90l[p,8,0]*ex_labels90l[p,10,0]>0:
                        ex_labels90l[p,9,0:2]=2*ex_labels90l[p,10,0:2]-ex_labels90l[p,8,0:2]
                        
    return ex_labels90l

# reverting midpoints depending on the body angle
def mid_point_reversion(angle_seq,path_saved,img_mask,u,labels,rot_params,crop_dic0,rot_angles,ref_width):
    
    jpegfiles = list()
    jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    mid_labels=np.zeros((labels.shape))
    mids_success=np.zeros((img_mask.shape))

    for t in range(len(jpegfiles)):
        z = fix_angle_extreams(angle_seq[t,1])

        # revert back coordinates based on angles
        if z==90 and img_mask[u-1,t]>0:
            skeleton=np.copy(labels[u-1,t,:,:])
            
            # create a temporary dictionary based on the rotational parameters 
            temp_dict=rot_params['im'+str(t).zfill(4)+'_in'+str(u)]
            zeroer=sum(np.transpose(skeleton))<15
            skeleton[:,0]/=256/temp_dict['sam1']
            skeleton[:,1]/=256/temp_dict['sam0']

            skeleton[:,0]+=temp_dict['sty']
            skeleton[:,1]+=temp_dict['stx']
            
            # scale coordinates based on the ratio the image was scaled
            skeleton *= 1/crop_dic0[str(t)][0]
            # transfered_labels[u-1,t,:,:] = skeleton
            
            mid_labels[u-1,t,:,:]=skeleton
            if sum(sum(skeleton[:,0:2]))==0:
                mids_success[u-1,t]=0
        
        elif z>90 and z!=180 and img_mask[u-1,t]>0:
            skeleton=np.copy(labels[u-1,t,:,:])

            temp_dict=rot_params['im'+str(t).zfill(4)+'_in'+str(u)]
            zeroer=sum(np.transpose(skeleton))<15
            skeleton[:,0]/=256/temp_dict['sam0']
            skeleton[:,1]/=256/temp_dict['sam1']
            

            skeleton[zeroer,:]=0
            
            # how to apply image flip to coordinates
            
            ind_greater = ~zeroer
            skeleton[ind_greater,1],skeleton[ind_greater,0]=skeleton[ind_greater,0] ,temp_dict['sam1']- skeleton[ind_greater,1]
            skeleton[ind_greater,0]=temp_dict['sam1']- skeleton[ind_greater,0]

            skeleton[:,0]+=temp_dict['sec_points'][0]
            skeleton[:,1]+=temp_dict['sec_points'][1]
            
            skeleton[zeroer,:]=0
            osize1 = temp_dict['size1']
            osize2 = temp_dict['size2']

            tr=rot_angles[str(int(abs(z)))]
            angle_r =tr*math.pi /12
            angle_r =tr*math.pi /12
            for uu in range(skeleton.shape[0]):
                if sum(skeleton[uu,:])>0:
                    skeleton[uu,0:2]=rot_back([osize1,osize2],np.array((temp_dict['im3'][0],temp_dict['im3'][1])),skeleton[uu,0:2],(angle_r))  

            
            skeleton[:,1] += temp_dict['sty']
            skeleton[:,0] += temp_dict['stx']
            
            skeleton *= 1/crop_dic0[str(t)][0]
            skeleton[:,0]=ref_width-skeleton[:,0]
            skeleton[zeroer,:]=0
            
            mid_labels[u-1,t,:,:]=skeleton
            if sum(sum(skeleton[:,0:2]))==0:
                mids_success[u-1,t]=0      
                

            
        elif z<90 and z!=0 and img_mask[u-1,t]>0:
            skeleton=np.copy(labels[u-1,t,:,:])

            temp_dict=rot_params['im'+str(t).zfill(4)+'_in'+str(u)]
            zeroer=sum(np.transpose(skeleton))<15
            skeleton[:,0]/=256/temp_dict['sam0']
            skeleton[:,1]/=256/temp_dict['sam1']
            
            
            skeleton[zeroer,:]=0
            
            # how to apply image flip to coordinates
            
            ind_greater = ~zeroer

            skeleton[ind_greater,0],skeleton[ind_greater,1]=skeleton[ind_greater,1] , temp_dict['sam0']- skeleton[ind_greater,0]

            skeleton[:,0]+=temp_dict['sec_points'][0]
            skeleton[:,1]+=temp_dict['sec_points'][1]
            
            
            skeleton[zeroer,:]=0
            osize1 = temp_dict['size1']
            osize2 = temp_dict['size2']

            tr=rot_angles[str(int(abs(z)))]
            angle_r =tr*math.pi /12
            for uu in range(skeleton.shape[0]):
                if sum(skeleton[u,:])>0:
                    skeleton[uu,0:2]=rot_back([osize1,osize2],np.array((temp_dict['im3'][0],temp_dict['im3'][1])),skeleton[uu,0:2],(angle_r))  

            temp_after_rot=np.copy(skeleton)
            skeleton[:,1] += temp_dict['sty']
            skeleton[:,0] += temp_dict['stx']
            
            skeleton[zeroer,:]=0

            
            skeleton *= 1/crop_dic0[str(t)][0]
            skeleton[zeroer,:]=0
        
            mid_labels[u-1,t,:,:]=skeleton
            if sum(sum(skeleton[:,0:2]))==0:
                mids_success[u-1,t]=0
                
    return mids_success,mid_labels

# align images using midpoints and saving them for implementing the main blazepose block
def align_images_midpoints(path_saved,final_mids,main_path,saving,rot_params,u,main_face):
    aa = time.time()
    jpegfiles = list()
    # jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    jpegfiles = sorted(main_face.keys())
    main_face_bp={}

    
    rot_params_bp={}
    
    for t in range(len(jpegfiles)):
        # I1=cv2.imread(path_saved+jpegfiles[t])
        I1 = main_face[jpegfiles[t]]
        skeleton=np.copy(final_mids[t])
        
        # if saving: here
        #     skeleton1=np.copy(final_mids[t]).astype(int)
        #     samasl=np.copy(I1)
        #     for ii in range(skeleton.shape[0]):
        #         if sum(skeleton1[ii][0:2])>5:
        #             cv2.circle(samasl, center=tuple(skeleton1[ii][0:2]), radius=5, color=(0, 0, 255), thickness=3)
        #     cv2.imwrite(main_path+'vid_test/temp/main/main_final_mids/final_mids_out_'+str(t).zfill(4)+'.jpeg',cv2.cvtColor(samasl, cv2.COLOR_BGR2RGB))
    
        
        zeroer=sum(np.transpose(skeleton))<10
        rot_info_1={}
        if skeleton[8,0]*skeleton[9,0]>0:
            mids_dist=np.linalg.norm(skeleton[8,0:2] - skeleton[9,0:2])
            if mids_dist>91:
                # first the image scaled down so that the mid points distant is less than 90 pixels considering the final size 
                # for feeding into blazepose is going to be 255
                ratio = 90/mids_dist
                
                # this ratio is applied to all coordinates
                wid_1=int(I1.shape[0]*ratio)
                len_1=int(I1.shape[1]*ratio)
                I1_resized=cv2.resize(I1,(len_1,wid_1))
                skeleton[:,0:2]=skeleton[:,0:2]*ratio
                temp_vec=[ratio,wid_1,len_1]
                
                # scaled paramaters are fed into align_im for cropping the image for blazepose
                img4,labels4,rot_info_1=align_im(I1_resized, np.copy(skeleton[8:10,0:2]))
                if img4.shape[0]*img4.shape[1]>10000:
                    temp_dict=rot_params['im'+str(t).zfill(4)+'_in'+str(u)]
                    rot_info_1['rot_vect']=temp_vec
                    rot_info_1['final_size']=img4.shape
                    
                    rot_params_bp['im'+str(t).zfill(4)]= rot_info_1
                    
                    # cv2.imwrite(main_path+'vid_test/temp/main/main_face_bp/face_out_'+str(t).zfill(4)+'.jpeg',img4)
                    main_face_bp['face_out_'+str(t).zfill(4)+'.jpeg']=img4
            else:
                I1_resized=I1
                temp_vec=[1,I1.shape[0],I1.shape[1]]
    # print('writing to disk:',time.time()-aa)
    return rot_params_bp,jpegfiles,main_face_bp

# filter and post process blazepose coordinates
def process_bp_coordinates(pre_processed_labels):
    
    ex_labels=np.zeros((pre_processed_labels.shape[0],pre_processed_labels.shape[3],3))
    for p in range(pre_processed_labels.shape[0]):   
        for q in range(pre_processed_labels.shape[3]):
            try_var=(pre_processed_labels[p,:,:,q])
            # res=try_var.argmax(axis=(1))
            ind = np.unravel_index(np.argmax(try_var, axis=None), try_var.shape)  # returns a tuple
            ex_labels[p,q,0:2]=np.array(list([ind[1],ind[0]]))*2
    return ex_labels
            
# bring back coorindates after finding coordinates with blazepose
def reverse_back_show_coord_after_bp(path_saved,ex_labels,rot_params_bp,saving,vect_dic,final_mids,out_ann,im_size0,im_size1,main_face):
    
    jpegfiles =list()
    jpegfiles = sorted(main_face.keys())
    yy=np.copy(ex_labels)
    over_all=np.zeros((len(jpegfiles),26,3))
    
    for t in range(len(jpegfiles)):
        # if saving:
        #     img = cv2.imread(path_saved+jpegfiles[t])
    
        try:
            skeleton = revert_back(np.copy(yy[t,:,0:2]),rot_params_bp['im'+str(t).zfill(4)]).astype(int)
            # if saving:
            #     for i in range(skeleton.shape[0]):
            #         cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=5)
            
            over_all[t,0:13,0:2]=skeleton
            
        except:
            pass
        
        try:
            
            vect=vect_dic[str(t)]
            temp_vect=np.copy(vect[0,4:-1].reshape((6,2)))
            temp_vect[:,0] *= im_size1
            temp_vect[:,1] *= im_size0
        
            skeleton=temp_vect.astype(int)
            # if saving:
            #     for ii in range(skeleton.shape[0]):
            #         cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(0, 0, 255), thickness=3)
            
            over_all[t,13:19,0:2]=skeleton
        except:
            pass
        
        try:
            
            skeleton=final_mids[t,:,:].astype(int)
            # if saving:
            #     for ii in range(4,skeleton.shape[0]):
            #         if ii!=7 and ii!=6:
            #             cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(255, 0, 0), thickness=3)
            
            # skeleton[6:8,:]=0
            over_all[t,19:,0:2]=skeleton[4:,0:2]
    
        except:
            pass
        
        # if saving:
        #     img=cv2.putText(img,"Full_Body, #F: "+str(t).zfill(4),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
    
            # out_ann.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    try:
        out_ann.release()
    except:
        pass
        
    return over_all

# There are multiple smoothing filters that can be applied to the sequence
# we should consider that the memory (sliding window) of this filters along time should be short otherwise, we can not
# implement them in realtime

# here I use a standard pandas interpolation

def smooth_over_all_coordinates_after_tracking(over_all_coordinates_with_tracking,out_ann,saving,path_saved,main_face):
    
    # read images
    over_all_smoothed_after_tracking = np.copy(over_all_coordinates_with_tracking)
    jpegfiles =list()
    # jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])

    jpegfiles = sorted(main_face.keys())
    
    # filter design
    order = 6
    fs = 20.0       # sample rate, Hz
    cutoff = 0.867  # desired cutoff frequency of the filter, Hz
    
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # # for x and y coordinates and all keypoints the interpolation is implemented as a time series filtering    
    # for i in range(0,over_all_coordinates_with_tracking.shape[1]):
    #     for j in range(0,2):
    #         temp_vec=np.copy(over_all_coordinates_with_tracking[:,i,j])
    #         temp_vec[temp_vec==0]=np.nan
    #         s = pd.Series( temp_vec)
    #         s_interpolated = s.interpolate()
    #         over_all_smoothed_after_tracking[:,i,j] = s_interpolated
            
    # # for x and y coordinates and all keypoints the interpolation is implemented as a time series filtering    
    for i in range(0,over_all_coordinates_with_tracking.shape[1]):
        if True: #i<13 or i>18:
            for j in range(0,2):
                data = over_all_coordinates_with_tracking[:,i,j]
                y = butter_lowpass_filter(data, cutoff, fs, order)
                y1 = np.copy(y)
                yy = y[-1]
    
                
                wh = 15
                y1 = y[wh:max(data.shape)].reshape((max(data.shape)-wh,1))
                for _ in range(0,wh):
                    y1 = np.concatenate((y1,yy.reshape((1,1))))
                
                y1[0:wh+10,0]=data[0:wh+10]
                y1[-wh:,0]=data[-wh:]
                
                            
                # temp_vec=np.copy(over_all_coordinates_with_tracking[:,i,j])
                # temp_vec[temp_vec==0]=np.nan
                # s = pd.Series( temp_vec)
                # s_interpolated = s.interpolate()
                over_all_smoothed_after_tracking[:,i,j] = y1[:,0]
            

    
    # for i in range(0,over_all_coordinates_with_tracking.shape[1]):
    #     for j in range(0,2):
    #         temp_vec=np.copy(over_all_coordinates_with_tracking[:,i,j])
    #         clean_signal = np.copy(temp_vec)
    #         # temp_vec[temp_vec==0]=np.nan
    #         # s = pd.Series( temp_vec)
    #         b = signal.firwin(999,0.1)
    #         f = fir1.Fir1(b)
    #         for ind in range(len(temp_vec)):
    #             clean_signal[ind] = f.filter(temp_vec[ind])
    #         # s_interpolated = filtfilt(b_2,1,temp_vec);
    #         # s_interpolated = s.interpolate()
    #         # over_all_smoothed_after_tracking[:,i,j] = s_interpolated
    
   
        
    # save data if parameter saving is set to true
    if saving:
        for t in range(len(jpegfiles)):
            img = main_face[jpegfiles[t]] # cv2.imread(path_saved+jpegfiles[t])
            # img_sk = np.copy(img)
            
            skeleton = over_all_smoothed_after_tracking[t,:].astype(int)

            
            hip_ind = [21,22,24]
            for u in hip_ind:
                if skeleton[u,0]<skeleton[25,0]:
                    skeleton[u,:]=0
                    if u==21 and skeleton[6,0]>10:
                        skeleton[u,:]=skeleton[6,:]
                    if u==22 and skeleton[9,0]>10:
                        skeleton[u,:]=skeleton[9,:]

            for ii in range(0,skeleton.shape[0]):
                if skeleton[ii,0]>20:
                    # cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(0, 255, 0), thickness=3)
                    if ii>12.5 and ii<19:
                        cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(0, 255, 0), thickness=2)
                    if ii in [0,1,2,6,7,8]:
                        cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(255, 0, 0), thickness=2)
                    if ii in [3,4,5,9,10,11]:
                        cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=3, color=(0, 0, 255), thickness=2)
                        
            
            bones=[(0,1),(1,2),(3,4),(4,5),(7,6),(7,8),(9,10),(10,11),(25,24),(6,9),(24,12)]
            for bone in bones:
                if skeleton[bone[0],0]*skeleton[bone[1],1]/100>0:
                    cv2.line(img, (skeleton[bone[0],0], skeleton[bone[0],1]), (skeleton[bone[1],0], skeleton[bone[1],1]), (0,0,255),thickness=2)
                    
            out_ann.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out_ann.release()
    try:
        out_ann.release()
    except:
        pass
    
    os.chdir("..")
    np.save("skeleton_after_smoothing",skeleton)
            
    return over_all_smoothed_after_tracking

