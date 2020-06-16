# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:37:11 2020

@author: Nikki
based on detect.py from https://github.com/hunglc007/tensorflow-yolov4-tflite
"""
import numpy as np
import tensorflow as tf
#import time
import cv2
from core.yolov4 import YOLOv4, decode #, YOLOv3_tiny, YOLOv3
#from absl import app, flags, logging
#from absl.flags import FLAGS
#from tensorflow.python.saved_model import tag_constants
from core import utils
from core.config import cfg
from PIL import Image
#from tensorflow import keras
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession


IMAGE_PATH = './data/kite.jpg'



def main():
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    WEIGHTS = './data/yolov4.weights'   #must end in .weights
    
    #probably change to uppercase
    INPUT_SIZE = 608
    

    
    'read image, convert to pillow colorspace, and find shape'
    original_image = cv2.imread(IMAGE_PATH)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    
    'resize image and add another dimension'
    image_data = utils.image_preprocess(np.copy(original_image), [INPUT_SIZE, INPUT_SIZE])   
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    'generate model'
    input_layer = tf.keras.Input([INPUT_SIZE, INPUT_SIZE, 3])
    
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, WEIGHTS)


    # model.summary()
    
    
    # # config1 = model.get_config()
    # #config2 = model.get_weights()
    # tf.keras.models.save_model(model,'my_model')
    # del model
    # #model.save_weights()
    # print ('thus far')
    # # loaded_model = keras.Model.from_config(config1)
    # #loaded_model.set_weights()
    # loaded_model = tf.keras.models.load_model('my_model')
    
    'make bboxes'
    pred_bbox = model.predict(image_data)
    #pred_bbox = model(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, INPUT_SIZE, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')

    'output bbox info to file and show image'
    utils.write_bbox_info(original_image, IMAGE_PATH, bboxes)
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()


    
if __name__ == "__main__":
    main()
