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
# def get_info(input_vids):
#     all_vid_info = [None]*len(input_vids)
#     for i, vid in enumerate(input_vids):
#         # video_path, GPS_pix, pix_GPS, origin
#         all_vid_info[i] = pg.sample_select(vid)
#     return all_vid_info

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
        print('Model built')
        
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
    # print(image_data.shape)
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

def batch_bboxes(model, frames):
    
    all_image_data = [None] * len(frames)
    sizes = [None] * len(frames)
    bbbb = [[],[]]
    
    for i, frame in enumerate(frames):
        #move frame to GPU
        all_image_data[i] = frame_to_gpu(frame)
        sizes[i] = frame.shape[:2]
    
    stacked = tf.stack(all_image_data, axis = 1)    
    
    trimmed = tf.squeeze(stacked)
    # print(trimmed.shape)
    
    # dataset = tf.data.Dataset.from_tensor_slices(stacked)
    # dataset = dataset.batch(2)
    # for im_data in dataset.as_numpy_iterator():
    pred_bbox = model.predict(trimmed)
    print('aww yeah')


    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    
    all_bboxes, probs, classes, image_nums = utils.postprocess_boxes(pred_bbox, sizes[0], INPUT_SIZE, 0.25)#.25
    bboxes = utils.filter_people(all_bboxes, probs, classes)
    bboxes = utils.nms(bboxes, 0.213, method='nms')
    bboxes = np.array(bboxes)
    
    # bboxes2 = utils.nms(bboxes2, 0.213, method='nms')
    # bboxes2 = np.array(bboxes2)
    
    print('frame1')
    # print(bboxes)
    # print('frame2')
    # print(bboxes2)
    
    
    # for i, im_data in enumerate(dataset.as_numpy_iterator()):
        
        
    #     pred_bbox = model.predict(im_data)
    
    #     pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
        
    #     all_bboxes, probs, classes = utils.postprocess_boxes(pred_bbox, sizes[0], INPUT_SIZE, 0.25)#.25
    #     bboxes = utils.filter_people(all_bboxes, probs, classes)
    
    #     #only continue processing if there were people identified
    #     if len(bboxes) > 0:
    #         #get rid of redundant boxes
    #         bbbb[i].append(utils.nms(bboxes, 0.213, method='nms')) #.213
    # # sep_bboxes = bbbb.unbatch()
    # # print(sep_bboxes)
    # print(bbbb)
    # return bbbb

###---------------------------------------------------------------------------
#   

def compliance_count(mytree, real_pts):
    errors = 0
    for pt in real_pts:
        dist, ind = mytree.query(pt, k=2)
        closest = mytree.data[ind[1]]
        #dist = pg.GPS_to_ft(gps_center, closest)
        if dist[1] < 6:
            errors = errors + 1
    return errors    

###---------------------------------------------------------------------------
#   Finds average distance occupants are apart from each other
###

def find_dist(mytree, real_pts):
    size = len(real_pts)
    # middle = size//2
    # med_dists = [None] * size
    avgs = [None] * size
    for i, pt in enumerate(real_pts):
        
        dist, _ = mytree.query(pt, size)
        # med_dists[i] = dist[middle]
        
        #do this for every pt in the tree - see if there is a built in function for this
        others = dist[1:]
        avgs[i] = sum(others)/len(others)
        
    # med_med = med_dists[middle]
    # avg_med = sum(med_dists)/len(med_dists)
    avg_avg = sum(avgs)/len(avgs)

    return avg_avg


# def find_occupants(frames, times, model, all_vid_info, files):

#     try:
#         all_image_data = [None] * len(frames)
#         results = [None] * len(frames)
#         for i, frame in enumerate(frames):
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#             #resize image and add another dimension
#             frame_size = frame.shape[:2]
#             cur_frame = np.copy(frame)
#             all_image_data[i] = utils.image_preprocess(cur_frame, [INPUT_SIZE, INPUT_SIZE]) 
#             all_image_data[i] = all_image_data[i][np.newaxis, ...].astype(np.float32)    
    
#         with tf.device('/GPU:0'):
#             all_image_data = tf.convert_to_tensor(all_image_data)    
#         #make bboxes
#         pred_bboxes = model.predict_on_batch(all_image_data)
#         for i, pred_bbox in enumerate(pred_bboxes):
#             f = files[i]
#             vid_info = all_vid_info[i]
#             pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
#             all_bboxes, probs, classes = utils.postprocess_boxes(pred_bbox, frame_size, INPUT_SIZE, 0.25)#.25
#             bboxes = utils.filter_people(all_bboxes, probs, classes)
#             #only continue processing if there were people identified
#             if len(bboxes) > 0:
#                 #get rid of redundant boxes
#                 bboxes = utils.nms(bboxes, 0.213, method='nms') #.213
                
#                 #draw bbox and get centered point at base of box
#                 frame = utils.draw_bbox(frame, bboxes, show_label = False)
#                 pts = utils.get_ftpts(bboxes)
                
#                 #draw radii and count people
#                 frame, six_count = pg.draw_radius(frame, pts, vid_info[1], vid_info[2], vid_info[3])
#                 people = pts.shape[0]
#             else:
#                 six_count = 0
#                 people = 0
#             #avg people and count within 6ft buffers   
            
#             dt = times[i]
#             #write info to file and overlay on video
#             utils.video_write_info(f, bboxes, str(dt), six_count, people)
#             utils.overlay_occupancy(frame, six_count, people, frame_size)
            
          
            
#             #convert frame to correct cv colors and display/record
#             results[i] = np.asarray(frame)
#             results[i] = cv2.cvtColor(results[i], cv2.COLOR_RGB2BGR)
#         return results
          
#     except:
#         print("Unexpected error:", sys.exc_info())
#         f.close()           


            
            
         