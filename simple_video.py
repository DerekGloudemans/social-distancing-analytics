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

#uncomment to verify that GPU is being used
tf.debugging.set_log_device_placement(True)
#tf.config.experimental.list_physical_devices('GPU') 


def main():
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    WEIGHTS = './data/yolov4.weights'   #must end in .weights
    video_path = './data/road.mp4' 
    #video_path = './data/ACC Export - 2020-03-02 11.46.34 AM.avi'# 
    #video_path = './data/vtest.avi'
    #video_path = './data/20190422_153844_DA4A.mkv'
    
    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)
    #vid.set(cv2.CAP_PROP_FPS, 10)
    
    # #probably change to uppercase
    INPUT_SIZE = 230
    
    'open file to output to'
    output_f = video_path[:-3] + 'txt'
    f = open(output_f, 'w')
    
    'generate model'
    #with strategy:
    input_layer = tf.keras.Input([INPUT_SIZE, INPUT_SIZE, 3])
    
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, WEIGHTS)
    try:
        
        while True:
            for i in range (10):
                vid.grab()
                
            return_value, frame = vid.read()
            dt = str(datetime.datetime.now())
            
            
            #check that the next frame exists
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                cv2.destroyWindow('result')
                print('Video has ended')
                break
            
            frame_size = frame.shape[:2]
            
            'resize image and add another dimension'
            image_data = utils.image_preprocess(np.copy(frame), [INPUT_SIZE, INPUT_SIZE])   
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            prev_time = time.time()
            
            'make bboxes'
            pred_bbox = model.predict(image_data)
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, INPUT_SIZE, 0.25)
            bboxes = utils.nms(bboxes, 0.213, method='nms')
            
            'output bbox info to file and show image'
        
            utils.video_write_info(frame, f, bboxes, dt)
            image = utils.draw_some_bbox(frame, bboxes)
            
            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" %(1000*exec_time)
            print(info)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR) #swapped image with result, not sure what the effect was
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    
        vid.release()
        cv2.destroyAllWindows()
        
        f.close()
            
    except:
        vid.release()
        cv2.destroyAllWindows()
        f.close()
    
    




    
if __name__ == "__main__":
    main()
