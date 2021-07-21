# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:53:05 2021

@author: Mohammad Ghahramani

import all required modules for Blazeface

"""
# global main_path
# main_path = 'c:/downloads/'

import numpy as np
import os
import tensorflow as tf
import pandas as pd


# main_path = np.load('settings.npy',allow_pickle=False)

main_pa=pd.read_csv('out.csv')
main_path=main_pa['path'][0]

os.chdir(main_path+'BlazePose')

from config import total_epoch, train_mode
from model import BlazePose

# import blazepose parameters for the shoulder and save them in model
model = BlazePose()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

# load pre-trained weights
checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_aic_LR_001_single_rot_11_face_dime/ckpt_{epoch}"
checkpoint_dir = os.path.dirname(checkpoint_path)
model.load_weights(checkpoint_path.format(epoch=145))


# import blazepose parameters for the body rotation and save them in model_rot

model_rot = BlazePose()
model_rot.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

# load pre-trained weights

checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_aic_LR_001_single_rot_11/ckpt_{epoch}"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_rot.load_weights(checkpoint_path.format(epoch=20))


# import blazepose parameters for the wholebody and save them in model_body

from model_13 import BlazePose as BP
model_body=BP()
model_body.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

# load pre-trained weights

checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_all3_hm_LR_001_13/ckpt_{epoch}"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_body.load_weights(checkpoint_path.format(epoch=8))

def read_data_for_midhip_extraction(path_saved,angle_seq,model_rot,vect_dic):
    jpegfiles = sorted([f for f in os.listdir(path_saved) if f.endswith('.jpeg')])
    data=np.zeros((angle_seq.shape[0],256,256,3))
    
    for t in range(len(jpegfiles)):
        vect=vect_dic[str(t)]
        
        # check if the face was detected
        if  vect.shape[0]>0:

            img = tf.io.read_file(path_saved+jpegfiles[t])
            img = tf.image.decode_image(img)
            data[t,...] = tf.image.resize(img, [256, 256])
    
    return model_rot.predict(data)

