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
import torch
import cv2
import time
import pandas as pd

# main_path = np.load('settings.npy',allow_pickle=False)

main_pa=pd.read_csv('out.csv')
main_path=main_pa['path'][0]

os.chdir(main_path+'BlazeFace')
from diagonal_crop.point import * 
from angular_cropping import angular_cropping
import diagonal_crop
from blazeface import BlazeFace

# setting params and vars obtained from blazeface standard github repo
# set up cpu usage instead of gpu using pytorch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize the net 
net = BlazeFace().to(gpu)

# initialize weights
net.load_weights("blazeface.pth")
net.load_anchors("anchors.npy")

# Optionally change the thresholds:
net.min_score_thresh = 0.75
net.min_suppression_threshold = 0.3

