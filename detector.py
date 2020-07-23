# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:44:39 2020

@author: Nikki
"""
import multiprocessing as mp
import addresses
import cv2
import sys
import datetime
import numpy as np
import time

import tensorflow as tf
from core.yolov4 import YOLOv4, decode #, YOLOv3_tiny, YOLOv3
from core import utils
from core.config import cfg
from PIL import Image
import pixel_gps as pg
import scipy.spatial

#uncomment to verify that GPU is being used
#tf.debugging.set_log_device_placement(True)

STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
XYSCALE = cfg.YOLO.XYSCALE
INPUT_SIZE = 419 #608 #230 #999 #800
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
WEIGHTS = './data/yolov4.weights'

###---------------------------------------------------------------------------
#   get transformations for camera locations and angles
#   
#   return - all_vid_info - array of arrays containing [video_path, GPS_pix, pix_GPS, origin] of input videos
def get_info(input_vids):
    all_vid_info = [None]*len(input_vids)
    for i, vid in enumerate(input_vids):
        # video_path, GPS_pix, pix_GPS, origin
        all_vid_info[i] = pg.sample_select(vid)
    return all_vid_info

###---------------------------------------------------------------------------
#   start people finding model
#
#   return - model - the object detection model
def start_model():

    tf.executing_eagerly()

    #TODO will have to change when working with several gpus
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    with strategy.scope():

        #generate model
        input_layer = tf.keras.Input([INPUT_SIZE, INPUT_SIZE, 3])
        
        feature_maps = YOLOv4(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, NUM_CLASS, i)
            bbox_tensors.append(bbox_tensor)    
        model = tf.keras.Model(input_layer, bbox_tensors)
        print('model built')
        
        #force to run eagerly
        model.run_eagerly = True
        
        #load existing weights into model
        utils.load_weights(model, WEIGHTS)
    
    return model
        

###---------------------------------------------------------------------------
#   

def frame_to_gpu(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cur_frame = np.copy(frame)
    im_data = utils.image_preprocess(cur_frame, [INPUT_SIZE, INPUT_SIZE]) 
    im_data = im_data[np.newaxis, ...].astype(np.float32)    
    
    with tf.device('/GPU:0'):
        im_data = tf.convert_to_tensor(im_data)

    return im_data

###---------------------------------------------------------------------------
#   

def person_bboxes(model, image_data, frame_size):
    #make bboxes
    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    all_bboxes, probs, classes = utils.postprocess_boxes(pred_bbox, frame_size, INPUT_SIZE, 0.25)#.25
    bboxes = utils.filter_people(all_bboxes, probs, classes)

    #only continue processing if there were people identified
    if len(bboxes) > 0:
        #get rid of redundant boxes
        bboxes = utils.nms(bboxes, 0.213, method='nms') #.213
    
    return bboxes



###---------------------------------------------------------------------------
#   

def compliance_count(gps_ftpts):
    errors = 0
    mytree = scipy.spatial.cKDTree(gps_ftpts)
    for gps_center in gps_ftpts:
        dist, ind = mytree.query(gps_center, k=2)
        closest = mytree.data[ind[1]]
        #dist = pg.GPS_to_ft(gps_center, closest)
        if dist[1] < 6:
            errors = errors + 1
    return errors    




def find_occupants(frames, times, model, all_vid_info, files):

    try:
        all_image_data = [None] * len(frames)
        results = [None] * len(frames)
        for i, frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            #resize image and add another dimension
            frame_size = frame.shape[:2]
            cur_frame = np.copy(frame)
            all_image_data[i] = utils.image_preprocess(cur_frame, [INPUT_SIZE, INPUT_SIZE]) 
            all_image_data[i] = all_image_data[i][np.newaxis, ...].astype(np.float32)    
    
        with tf.device('/GPU:0'):
            all_image_data = tf.convert_to_tensor(all_image_data)    
        #make bboxes
        pred_bboxes = model.predict_on_batch(all_image_data)
        for i, pred_bbox in enumerate(pred_bboxes):
            f = files[i]
            vid_info = all_vid_info[i]
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            all_bboxes, probs, classes = utils.postprocess_boxes(pred_bbox, frame_size, INPUT_SIZE, 0.25)#.25
            bboxes = utils.filter_people(all_bboxes, probs, classes)
            #only continue processing if there were people identified
            if len(bboxes) > 0:
                #get rid of redundant boxes
                bboxes = utils.nms(bboxes, 0.213, method='nms') #.213
                
                #draw bbox and get centered point at base of box
                frame = utils.draw_bbox(frame, bboxes, show_label = False)
                pts = utils.get_ftpts(bboxes)
                
                #draw radii and count people
                frame, six_count = pg.draw_radius(frame, pts, vid_info[1], vid_info[2], vid_info[3])
                people = pts.shape[0]
            else:
                six_count = 0
                people = 0
            #avg people and count within 6ft buffers   
            
            dt = times[i]
            #write info to file and overlay on video
            utils.video_write_info(f, bboxes, str(dt), six_count, people)
            utils.overlay_occupancy(frame, six_count, people, frame_size)
            
          
            
            #convert frame to correct cv colors and display/record
            results[i] = np.asarray(frame)
            results[i] = cv2.cvtColor(results[i], cv2.COLOR_RGB2BGR)
        return results
          
    except:
        print("Unexpected error:", sys.exc_info())
        f.close()           


            
            
         