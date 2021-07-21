# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 17:16:24 2021

@author: Mohammad Ghahramani

"""

from argparse import ArgumentParser
from blazepose_pipeline_inference_fn import *
import cv2
import os


def main():
    parser = ArgumentParser()
    parser.add_argument("--video_path", default="vid_test/y2mate9.mp4", help="Path to the video")
    parser.add_argument("--frames", default=100, help="number of frames to process")
    
    args = parser.parse_args()
    blazepose_inf(os.getcwd()+'\\',args.video_path,args.frames)
    
if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()

