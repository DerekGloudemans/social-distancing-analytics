# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:25:10 2020

@author: Nikki
"""

import importlib
import multiprocessing as mp
# import addresses
import ip_streamer
import cv2
import sys
# import datetime
import numpy as np
import time
from ctypes import c_bool
import scipy.spatial

# import tensorflow as tf
# from core.yolov4 import YOLOv4, decode #, YOLOv3_tiny, YOLOv3
from core import utils
# from core.config import cfg
# from PIL import Image
import pixel_realworld as pr
import detector
import transform as tform
import csv
import ast
import analyze_data as adat
#uncomment to verify that GPU is being used
#tf.debugging.set_log_device_placement(True)

######## FIXMEEEE - analyze data needs a way to access shared variable, must make q more easily shareable
########
########
########
########


def main():
    importlib.reload(mp)
    # ips = ['C:/Users/Nikki/Documents/work/inputs-outputs/video/AOTsample1_1.mp4','C:/Users/Nikki/Documents/work/inputs-outputs/video/AOTsample2_1.mp4' ]
    ips = []
    vids = []
    # file containing camera information
    transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/transforms.csv'
    transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/test.csv'
    transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/aot_transforms_better.csv'
    
    manager = mp.Manager()
    #create VidObjs to store information about each camera
    initialize_cams(transform_f, ips, vids)

    #  create manager to handle shared variables across processes
    updated = manager.Value(c_bool, False)
    removed = manager.Value(c_bool, False)

    num_cams = len(vids)
    frames = manager.list([None]* num_cams)
    times = manager.list([None]* num_cams)
    avgs = manager.list([None] * 5)
    avg_lock = manager.Lock()
    out_q = manager.Queue(num_cams*2)
    
    #list of queues, one queue per camera, that contain lists of output data (occupants, errors, avg dist)
    all_output_stats = []
    #length of queues, kinda arbitrary - this is the number that will be used for moving avg analytics
    buf_num = 3
    # for i in range(num_cams):
    #     all_output_stats.append(mp.Queue(buf_num))        
    
    
    #stores frame data that has been transfered to GPU
    gpu_frames = [None]* num_cams        

    #start model
    model = detector.start_model()

    try:
        #grab video frames in separate process
        streamer = mp.Process(target=ip_streamer.stream_all, args=(frames, times, ips, updated)) 
        streamer.start()
        print('Separate process started')
        
        # analysis = mp.Process(target=adat.main, args=(all_output_stats, buf_num, avgs, removed))
        analysis = mp.Process(target=adat.main, args=(out_q, buf_num, num_cams, avgs, avg_lock)) 
        analysis.start()
        print('Separate process started')
        
        #wait until frames are starting to be read
        while(not updated.value):   
            continue
        
        #find and assign the frame size of each stream
        #TODO may have to resize frames or something for running through the model in batches
        for i, vid in enumerate(vids):
            vid.set_size(frames[i].shape[:2])
            
        prev_time = time.time()
        
        #continuously loop until keyboard interrupt
        while(True):
            
            curr_time = time.time()
            
            # save outputs every 5 minutes
            if (curr_time - prev_time) > (5 * 60):
                save_files(vids)
                prev_time = curr_time
                
            
            # verify that frame is not the same as it was last time it was displayed
            if updated.value == True:
                 # detector.batch_bboxes(model, frames)
                # detector.batch_bboxes(model, frames)
                    
                #loop through frames and move to GPU (TODO - enable batching)
                for i, frame in enumerate(frames):
                    gpu_frames[i] = detector.frame_to_gpu(frame)
            
                #loop through frames, find people, record occupants and infractions  
                for i, vid in enumerate(vids): 
                    frame = frames[i]
                    gpu_frame = gpu_frames[i]
                    dt = times[i]
                    bboxes = detector.person_bboxes(model, gpu_frame, vid.frame_size)
                    
                    #find ft pts and convert to real_world
                    ftpts = utils.get_ftpts(bboxes)
                    realpts = tform.transform_pt_array(ftpts, vid.pix_real)
                    
                    # verifies there is more than one p[oint in the list (each point has size 2)]
                    if realpts.size > 2:
                        mytree = scipy.spatial.cKDTree(realpts)
                        errors = detector.compliance_count(mytree, realpts)
                        avg_dist = detector.find_dist(mytree, realpts)
                    else:
                        errors = 0
                        avg_dist = None
                    occupants = ftpts.size//2
                    
                    #output info to csv file  
                    utils.video_write_info(vid.csvfile, realpts, str(dt), errors, occupants, avg_dist)
                    
                    stats = [i, errors, occupants, avg_dist]
                    
                    #put outpt data into queue so it is accessible by the analyzer
                    # print(list(all_output_stats))
                    # print(repr(all_output_stats))
                    # print(list(all_output_stats[i]))
                    if out_q.full():
                        a = out_q.get()
                        print("Leaving queue: ", a)
                    out_q.put(stats)
                    # if all_output_stats[i].full():
                    #     a = all_output_stats[i].get()
                    #     removed.value = True
                    #     print("Leaving queue: ", a)
                    # else:
                    #     removed.value = False
                    # all_output_stats[i].put(stats)
                    
                    avg_lock.acquire()
                    print("Avgs: ", avgs)
                    avg_lock.release()
                    #save frames
                    if vid.frame_save:
                        outpt_frame(ftpts, frame, vid, errors, occupants, bboxes)
                        
                    # FIXME - just for debugging, show frame on screen
                    if i == 0 or i == 1:
                        show_frame(ftpts, frame, vid, errors, occupants, bboxes, i)  
                        if cv2.waitKey(1) & 0xFF == ord('q'): break
           
    except:
        print("Unexpected error:", sys.exc_info())
        cv2.destroyAllWindows()
        for vid in vids:
            vid.base_f.close()
        streamer.terminate()
        analysis.terminate()


###---------------------------------------------------------------------------
#   Saves frame with overlaid info
          
def outpt_frame(ftpts, frame, vid, errors, occupants, bboxes):
    result = prep_frame(ftpts, frame, vid, errors, occupants, bboxes)

    cv2.imwrite(vid.frame_dir + str(vid.count) + '.jpg', result)
    vid.count = vid.count + 1
  
    
###---------------------------------------------------------------------------
#   Shows frame with overlaid info
          
def show_frame(ftpts, frame, vid, errors, occupants, bboxes, i):
    result = prep_frame(ftpts, frame, vid, errors, occupants, bboxes)

    cv2.namedWindow("result" + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("result" + str(i), result) 
    
        

###---------------------------------------------------------------------------
#   Overlays ellipses, bboxes, stats on frame
 
def prep_frame(ftpts, frame, vid, errors, occupants, bboxes):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(bboxes) > 0:
        frame = utils.draw_bbox(frame, bboxes, show_label = False)
        frame, x = pr.draw_radius(frame, ftpts, vid.real_pix, vid.pix_real, vid.origin)
    utils.overlay_occupancy(frame, errors, occupants, vid.frame_size)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return result


###---------------------------------------------------------------------------
#   Closes and reopens csvfiles to save them
   
def save_files(vids):
    for vid in vids:
        vid.base_f.close()
        vid.base_f = open(vid.filename, 'a', newline='')
        vid.csvfile = csv.writer(vid.base_f)
        print('Saved')


###---------------------------------------------------------------------------
#   Creates VidObjs from addresses listed in csvfile
   
def initialize_cams(transform_f, ips, vids):
    with open(transform_f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            try:
                vid_path = row[0]
                ips.append(vid_path)
                pixr = row[1]
                pix_real = ast.literal_eval(pixr)
                realp = row[2]
                real_pix = ast.literal_eval(realp)
            
                # pix_real = np.array(pix_real)
                if vid_path[:3] == 'C:/':
                    fragments = vid_path.split('/')
                    end = fragments[-1]
                    extended = end.split('.')
                    name = extended[0]
                    print('Video ' + str(i +1) + ' properly initialized: ' + name)
                elif vid_path[:7] == 'rtsp://':
                    fragments = vid_path.split('@')
                    end = fragments[1]
                    extended = end.split('/')
                    name = extended[0]
                    print('IP cam ' + str(i +1) + ' properly initialized: ' + name)
                else:
                    print('Invalid camera name')
                
                vids.append(VidObj(name, pix_real, real_pix)) #True))
                print()
                
            except:
                print('Cam ' + str(i + 1) + ' FAILED initialization')
                print()
                
###---------------------------------------------------------------------------
#       

class VidObj:
    
    def __init__(self, name, pix_real, real_pix, save = False):
        #name of output file
        self.filename = 'C:/Users/Nikki/Documents/work/inputs-outputs/txt_output/' + name + '.csv'
        
        #create base file and csv writer to add to file
        self.base_f = open(self.filename, 'a', newline='')
        self.csvfile = csv.writer(self.base_f)
        
        #set whether or not to save file
        self.frame_save = save
        
        #set transformation matrices
        self.pix_real = pix_real
        self.real_pix = real_pix
        self.origin = np.array([0,0])
        # self.transform_info = pr.sample_select(name)
        
        #set output directory and frame number in case video is to be saved
        self.frame_dir = 'C:/Users/Nikki/Documents/work/inputs-outputs/vid_output/' + name + '_frames/'
        self.count = 0  
                
        print('Saving frames: ', self.frame_save)
        
        
    def set_size(self, frame_size):
        self.frame_size = frame_size
        self.origin = np.array([frame_size[1]/2, frame_size[0]])
        
    def start_save(self):
        self.frame_save = True
        
    def end_save(self):
        self.frame_save = False
        
            
if __name__ == '__main__':
    main()