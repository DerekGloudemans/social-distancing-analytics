# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:25:10 2020

@author: Nikki
"""

import importlib
import multiprocessing as mp
import addresses
import ip_streamer
import cv2
import sys
import datetime
import numpy as np
import time
from ctypes import c_bool

import tensorflow as tf
from core.yolov4 import YOLOv4, decode #, YOLOv3_tiny, YOLOv3
from core import utils
from core.config import cfg
from PIL import Image
import pixel_realworld as pg
import detector
import transform as tform
import csv
#uncomment to verify that GPU is being used
#tf.debugging.set_log_device_placement(True)




def main():
    importlib.reload(mp)
    #initialize bufefers, addresses, multiprocessing manager, etc
    ips = addresses.AOT
    buf_len = len(ips)
    manager = mp.Manager()
    frames = manager.list([None]* buf_len)
    times = manager.list([None]* buf_len)
    updated = manager.Value(c_bool, False)
    
    files = [None]* buf_len
    base_f = [None]* buf_len
    all_image_data = [None]* buf_len
    frame_sizes = [None]* buf_len
    
    cam1 = VidObj('aot1', False)
    cam2 = VidObj('aot2', False)
    
    vids = [cam1, cam2]
    
    #get transformation and origin for all video locations
    # all_vid_info = detector.get_info(['aot1', 'aot2'])
    
    #start model
    model = detector.start_model()
    
    
    # #open output files
    # for i in range(buf_len):
    #     output_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/txt_output/mp_test' + str(i) +'.csv'
    #     base_f[i] = open(output_f, 'w', newline='')
    #     files[i] = csv.writer(base_f[i])
    #     print('file started')
        
    try:
        #start process to grab video frames
        streamer = mp.Process(target=ip_streamer.stream_all, args=(frames, times, ips, updated)) 
        streamer.start()
        #continuously loop until keyboard interrupt
        while(not updated.value):   
            continue

        for i, vid in enumerate(vids):
            vid.set_size(frames[i].shape[:2])
        while(True):
            #prev_time = time.time()
            #verify that frame is not the same as it was last time it was displayed
            #if updated.value == True:
            #loop through frames and move to GPU (TODO - enable batching)

            for i, frame in enumerate(frames):
                
                #move frame to GPU
                all_image_data[i] = detector.frame_to_gpu(frame)
                #frame_sizes[i] = frame.shape[:2]
            #loop through frames, find people, record occupants and infractions  
            for i, frame in enumerate(frames): 
                vid_info = vids[i].transform_info
                
                #need bbox locations if planning to display on each frame
                #bboxes = detector.person_bboxes(model, all_image_data[i], frame_sizes[i])
                bboxes = detector.person_bboxes(model, all_image_data[i], vids[i].frame_size)
                #find ft pts and convert to GPS
                ftpts = utils.get_ftpts(bboxes)
                gps = tform.transform_pt_array(ftpts, vid_info[2])
                if gps.size > 2:
                    errors = detector.compliance_count(gps)
                else:
                    errors = 0
                occupants = ftpts.shape[0]
                #output info to file  
                utils.video_write_info(vids[i].csvfile, gps, str(times[i]), errors, occupants)

                if vids[i].frame_save:
                    outpt_frame(ftpts, frame, vids[i], errors, occupants, bboxes)
                if i == 0:
                    result, overhead = show_frame(ftpts, frame, vids[i], errors, occupants, bboxes)
                    cv2.namedWindow("Overhead", cv2.WINDOW_NORMAL)
                    cv2.imshow("Overhead", overhead)
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", result)   

                if cv2.waitKey(1) & 0xFF == ord('q'): break
            # curr_time = time.time()
            # exec_time = curr_time - prev_time
            # info = "time: %.2f ms" %(1000*exec_time/len(ips))
            # print(info)
                    
    except:
        print("Unexpected error:", sys.exc_info())
        cv2.destroyAllWindows()
        for vid in vids:
            vid.base_f.close()
        streamer.terminate()
        
def outpt_frame(ftpts, frame, vid, errors, occupants, bboxes):
    vid_info = vid.transform_info
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(bboxes) > 0:
        frame = utils.draw_bbox(frame, bboxes, show_label = False)
        frame, x = pg.draw_radius(frame, ftpts, vid_info[1], vid_info[2], vid_info[3])
    utils.overlay_occupancy(frame, errors, occupants, vid.frame_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite(vid.frame_dir + str(vid.count) + '.jpg', frame)
    vid.count = vid.count + 1

def show_frame(ftpts, frame, vid, errors, occupants, bboxes):
    vid_info = vid.transform_info
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(bboxes) > 0:
        frame = utils.draw_bbox(frame, bboxes, show_label = False)
        frame, bird_img, x = pg.draw_radius(frame, ftpts, vid_info[1], vid_info[2], vid_info[3])
    utils.overlay_occupancy(frame, errors, occupants, vid.frame_size)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return result, bird_img

     
class VidObj:
    def __init__(self, name, save):
        output_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/txt_output/' + name + '.csv'
        self.base_f = open(output_f, 'w', newline='')
        self.csvfile = csv.writer(self.base_f)
        print('file started')
        
        self.frame_save = save
        self.transform_info = pg.sample_select(name)
        self.frame_dir = 'C:/Users/Nikki/Documents/work/inputs-outputs/vid_output/' + name + '_frames/'
        self.count = 0  
        
        print('Name: ', name)
        print('Saving: ', self.frame_save)
        
    def set_size(self, frame_size):
        self.frame_size = frame_size
            
if __name__ == '__main__':
    main()