#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to a dir with images')

argparser.add_argument(
    '-r',
    '--runID',
    help='runID')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    runID   = args.runID

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################
    outputdir = os.path.join('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/YOLO/false_positives_dir',runID)
    if os.path.exists(outputdir):
        print('path exists')
    else:
        os.makedirs(outputdir)
    print(image_path)
    imlist = os.listdir(image_path)
    print(len(imlist))
    for file in imlist:
        if file.endswith('jpg'):
            print(file)
            image = cv2.imread(os.path.join(image_path,file))
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])
            print(len(boxes), 'boxes are found')
            if len(boxes)>0:
                filename = file[:-4] + '_detected' + file[-4:]
                output = os.path.join(outputdir,filename)
                print(output)
                cv2.imwrite(output,image)
#                command = 'cp %s %s' %(os.path.join('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/BirdsAI_random/to_test/',file),os.path.join('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/BirdsAI_random/tested_with_boxes/',file))
#                os.system(command)
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
