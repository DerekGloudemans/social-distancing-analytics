# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:03:56 2020

@author: Nikki
based on detect.py and detectvideo.py from https://github.com/hunglc007/tensorflow-yolov4-tflite
"""


import numpy as np
import tensorflow as tf
import time
import cv2
from core.yolov4 import YOLOv4, decode #, YOLOv3_tiny, YOLOv3
#from absl import app, flags, logging
#from absl.flags import FLAGS
#from tensorflow.python.saved_model import tag_constants
from core import utils
from core.config import cfg
from PIL import Image
import datetime
#from tensorflow import keras
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
import sys
from threading import Thread
from queue import Queue
import pixel_gps as pg
#uncomment to verify that GPU is being used
#tf.debugging.set_log_device_placement(True)

#@tf.function

def main():
    tf.executing_eagerly()
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    with strategy.scope():
    # if True:
    
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
        XYSCALE = cfg.YOLO.XYSCALE
        WEIGHTS = './data/yolov4.weights'   #must end in .weights
        video_path = './data/road.mp4' 
        video_path = './data/AOTsample3.mp4' 
        #video_path = './data/vtest.avi'
        #video_path = './data/20190422_153844_DA4A.mkv'
        
        print("Video from: ", video_path )
        vid = cv2.VideoCapture(video_path)
        
        GPS_pix, pix_GPS = pg.get_transform()
        
        
        print('thread started')
        INPUT_SIZE = 419 #608 #230
        #open file to output to
        output_f = video_path[:-3] + 'txt'
        f = open(output_f, 'w')
        print('file started')

        
        #generate model
        input_layer = tf.keras.Input([INPUT_SIZE, INPUT_SIZE, 3])
        print('tensors started 1')
        feature_maps = YOLOv4(input_layer, NUM_CLASS)
        print('tensors started 2')
        bbox_tensors = []
        print('tensors started 3')

        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, NUM_CLASS, i)
            bbox_tensors.append(bbox_tensor)
        print('tensors started 4')
        model = tf.keras.Model(input_layer, bbox_tensors)
        print('model built')
        
        
        #force to run eagerly
        model.run_eagerly = True
        if model.run_eagerly:
            print ('yeeyee')
        else:
            print ('hawhaw')
        utils.load_weights(model, WEIGHTS)
 
        #continue reading and showing frames until interrupted
        try:
            
            while True:
                #skip desired number of frames to speed up processing
                for i in range (5):
                    vid.grab()
                
                #get current time - when using video streams, will be correct
                dt = str(datetime.datetime.now())    
                return_value, frame = vid.read()
                
                # check that the next frame exists
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    cv2.destroyWindow('result')
                    print('Video has ended')
                    break
                frame_size = frame.shape[:2]
                
                #resize image and add another dimension
                cur_frame = np.copy(frame)
                
                image_data = utils.image_preprocess(cur_frame, [INPUT_SIZE, INPUT_SIZE]) 
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                
                prev_time = time.time()
                
                with tf.device('/GPU:0'):
                    image_data = tf.convert_to_tensor(image_data)
                    print(image_data.device)
                    
                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = "time1: %.2f ms" %(1000*exec_time)
                print(info)
                prev_time = time.time()
                
                
                #make bboxes
                pred_bbox = model.predict(image_data)
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
                bboxes = utils.postprocess_boxes(pred_bbox, frame_size, INPUT_SIZE, 0.25)
                bboxes = utils.nms(bboxes, 0.213, method='nms')
                
                
                #output bbox info to file and show image
                #calculate and display time it took to process frame
                utils.video_write_info(frame, f, bboxes, dt)
                image, pts = utils.draw_some_bbox(frame, bboxes)
                image = pg.draw_radius(image, pts, GPS_pix, pix_GPS)
                
                
                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = "time2: %.2f ms" %(1000*exec_time)
                print(info)
                
                result = np.asarray(image)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR) #swapped image with result, not sure what the effect was
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
            #end video, close viewer, stop writing to file
            vid.release()
            cv2.destroyAllWindows()
            f.close()
        
        
        #if interrupted, end video, close viewer, stop writing to file
        except:
            print("Unexpected error:", sys.exc_info()[0])
            vid.release()
            cv2.destroyAllWindows()
            f.close()
 
        
        
        
if __name__ == "__main__":
    main()
