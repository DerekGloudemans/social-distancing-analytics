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
#from threading import Thread
#from queue import Queue
import pixel_gps as pg
#uncomment to verify that GPU is being used
#tf.debugging.set_log_device_placement(True)

#@tf.function

def main():
    
    #not sure whether this is effective or not
    tf.executing_eagerly()
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    with strategy.scope():
    # if True:
        
        #SETTINGS TO ADJUST         
        #whether or not to save video to output file or show on screen
        RECORD = True
        INPUT_VID = 'aot1'
        #INPUT_VID = 'mrb3'
        #INPUT_VID
        OUTPUT_VID= './data/vid_output/' + INPUT_VID + '.avi'
        SHOW_VID = False
        THROWOUT_NUM = 3           #min is 1
        INPUT_SIZE = 419 #608 #230 #999 #800
        
        #initialize constants
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
        XYSCALE = cfg.YOLO.XYSCALE
        WEIGHTS = './data/yolov4.weights'   #must end in .weights



        #setup variables based on what video is being used
        video_path, GPS_pix, pix_GPS, origin = pg.sample_select(INPUT_VID)
        
        #start video capture
        print("Video from: ", video_path )
        vid = cv2.VideoCapture(video_path)
        
        #initialize occupancy and compliance buffers
        buf_size = 5
        count_buf = buf_size * [0]
        ind = 0
        people_buf = buf_size * [0]
        
        #open file to output to
        output_f = './data/txt_output/' + INPUT_VID + '.txt'
        f = open(output_f, 'w')
        print('file started')
        f.write('Time\t\t\t\tPed\t<6ft\n')
        
        #define writer and output video properties
        if RECORD:
            fps = vid.get(5)
            wdt = int(vid.get(3))
            hgt = int(vid.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_vid = cv2.VideoWriter(OUTPUT_VID, fourcc, fps/THROWOUT_NUM, (wdt, hgt))


        
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
 
        #continue reading and showing frames until interrupted
        try:
            while True:
                
                #skip desired number of frames to speed up processing
                for i in range (THROWOUT_NUM):
                    vid.grab()
                
                #get current time and next frame
                dt = str(datetime.datetime.now())    
                return_value, frame = vid.retrieve()
                
                # check that the next frame exists, if not, close display window and exit loop
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #image = Image.fromarray(frame)
                else:
                    if SHOW_VID:
                        cv2.destroyWindow('result')
                    print('Video has ended')
                    break
                
                #resize image and add another dimension
                frame_size = frame.shape[:2]
                cur_frame = np.copy(frame)
                image_data = utils.image_preprocess(cur_frame, [INPUT_SIZE, INPUT_SIZE]) 
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                
               
                prev_time = time.time()  #for calculating how long it takes to process a frame
                
                
                with tf.device('/GPU:0'):
                    image_data = tf.convert_to_tensor(image_data)
                    print(image_data.device)
                
                #for calculating how long it takes to process a frame
                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = "time1: %.2f ms" %(1000*exec_time)
                print(info)
                prev_time = time.time()
                
                #make bboxes
                pred_bbox = model.predict(image_data)
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
                bboxes = utils.postprocess_boxes(pred_bbox, frame_size, INPUT_SIZE, 0.25)#.25
    
                #only continue processing if there were people identified
                if len(bboxes) > 0:
                    #get rid of redundant boxes
                    bboxes = utils.nms(bboxes, 0.213, method='nms') #.213
                    
                    #draw bbox and get centered point at base of box
                    frame = utils.draw_bbox(frame, bboxes, show_label = False)
                    pts = utils.get_ftpts(bboxes)
                    
                    #draw radii and count people
                    frame, count_buf[ind] = pg.draw_radius(frame, pts, GPS_pix, pix_GPS, origin)
                    people_buf[ind] = pts.shape[0]
                else:
                    count_buf[ind] = 0
                    people_buf[ind] = 0
                    
                #avg people and count witihin 6ft buffers   
                people = int(sum(people_buf)/len(people_buf))
                count = int(sum(count_buf)/len(count_buf))
                
                #write info to file and overlay on video
                utils.video_write_info(f, bboxes, dt, count, people)
                utils.overlay_occupancy(frame, count, people, frame_size)
                
                #for calculating how long it takes to process a frame
                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = "time2: %.2f ms" %(1000*exec_time)
                print(info)
                
                #convert frame to correct cv colors and display/record
                result = np.asarray(frame)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                
                if SHOW_VID:
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", result)
                if RECORD:
                    out_vid.write(result)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                #increment index
                ind = (ind + 1) % buf_size
                
            #end video, close viewer, stop writing to file
            vid.release()
            if RECORD:
                out_vid.release()
            if SHOW_VID:
                cv2.destroyAllWindows()
            f.close()
            
        #if interrupted, end video, close viewer, stop writing to file
        except:
            print("Unexpected error:", sys.exc_info()[0])
            vid.release()
            if RECORD == True:
                out_vid.release()
            if SHOW_VID:
                cv2.destroyAllWindows()
            f.close()


if __name__ == "__main__":
    main()
