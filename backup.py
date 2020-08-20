# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:47:04 2020

@author: Nikki
"""



import importlib
import multiprocessing as mp
import ip_streamer
import cv2
import sys
import numpy as np
import time
from ctypes import c_bool
import scipy.spatial

import tensorflow as tf
from core import utils
import pixel_realworld as pr
import detector
import transform as tform
import csv
import ast
import analyze_data as adat



######## FIXMEEEE - analyze data needs a way to access shared variable, must make q more easily shareable
########
########
########
########


def main():
    #length of queues, kinda arbitrary - this is the number that will be used for moving avg analytics
    buf_num = 3
    
    #list of workers
    
    
    #uncomment to verify that GPU is being used
    tf.debugging.set_log_device_placement(False)
    importlib.reload(mp)

    ips = []
    vids = []
    
    # file containing camera information
    transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/transforms.csv'
    transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/test.csv'
    transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/test_all.csv'
    transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/aot_transforms_better.csv'
    
    manager = mp.Manager()
    
    #create VidObjs to store information about each camera
    initialize_cams(transform_f, ips, vids)
    
    num_cams = len(vids)
    
    #  create manager to handle shared variables across processes
    updated = manager.Value(c_bool, False)
    frames = manager.list([None]* num_cams)
    times = manager.list([None]* num_cams)
    avgs = manager.list([None] * 5)
    
    
    errors = manager.list()
    occupants = manager.list()
    avg_dists = manager.list()
    
    
    e_lock = manager.Lock()
    o_lock = manager.list()
    d_lock = manager.list()
    avg_lock = manager.Lock()
    out_q = manager.Queue(num_cams*2)
    
    
    
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
        analysis = mp.Process(target=adat.main, args=(out_q, buf_num, num_cams, avgs, avg_lock, errors, occupants, avg_dists, e_lock, o_lock, d_lock)) 
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
                        errs = detector.compliance_count(mytree, realpts)
                        avg_dist = detector.find_dist(mytree, realpts)
                    else:
                        errs = 0
                        avg_dist = None
                    ocpts = ftpts.size//2
                    
                    #output info to csv file  
                    utils.video_write_info(vid.csvfile, realpts, str(dt), errs, ocpts, avg_dist)
                    
                    stats = [i, errs, ocpts, avg_dist]
                    
                    #put outpt data into queue so it is accessible by the analyzer
                    if out_q.full():
                        a = out_q.get()
                        print("Leaving queue: ", a)
                    out_q.put(stats)
                    # if len(avg_dists) > 0:         
                    #     print (adat.get_o_avg(occupants, i))
                    #     print (adat.get_e_avg(errors, i))
                    #     print (adat.get_dist_avg(avg_dists, i))
                    #     print()
                    avg_lock.acquire()
                    print("Avgs: ", avgs)
                    avg_lock.release()
                    #save frames
                    if vid.frame_save:
                        outpt_frame(ftpts, frame, vid, errs, ocpts, bboxes)
                        
                    # FIXME - just for debugging, show frame on screen
                    show_frame(ftpts, frame, vid, errs, ocpts, bboxes, i)  
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
           
    except:
        print("Unexpected error:", sys.exc_info())
        cv2.destroyAllWindows()
        for vid in vids:
            vid.base_f.close()
        streamer.terminate()
        analysis.terminate()
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:35:56 2020

@author: Nikki
"""
# 1 website only displays live videos and csvs of 1 cam
# 2 website displays live videos + csvs of all
# 3 website displays live individual camera and live aggregate data
# 4 website shows most recent frames
# 5 website shows moving averages + such for hr, day, week
# 6 website shows graphs and such
import sys
import math

#TODO I think averages are correct, could be good to double check
def main(out_q, buf_num ,num_cams, avgs, avg_lock, errors, occupants, avg_dists, e_lock, o_lock, d_lock):
    # all_out = [[[None]*3]* buf_num] * num_cams
    e_lock.acquire()
    o_lock.acquire()
    d_lock.acquire()
    errors = errors.append( [[None]* buf_num for _ in range(num_cams)])
    occupants = occupants.append([[None]* buf_num for _ in range(num_cams)])
    avg_dists = avg_dists.append([[None]* buf_num for _ in range(num_cams)])
    e_lock.release()
    o_lock.release()
    d_lock.release()

    index = 0
    rollover = 0
    
    #not sure that a is cycling through the buffers in the best way
    try:
        while(True):
            #location of oldest entry in mega list
            index = index % buf_num
            if not out_q.empty():
                # total = total+1
                rollover = rollover + 1
                data = out_q.get()
                #camera number
                i = data[0]
               
                #updates camera buffer at oldest index
                with e_lock:
                    errors[i][index] = data[1]
                with o_lock:
                    occupants[i][index] = data[2]
                with d_lock:
                    avg_dists[i][index] = data[3]
                
                if rollover == (num_cams):
                    rollover = 0
                    index = index + 1

                
                
                
                
                #this section just for debugging
                avg_lock.acquire()
                avgs[0] = i
                avgs[1] = index
            
                avgs[2] = get_o_avg(occupants, i)
                avgs[3] = get_e_avg(errors, i)
                avgs[4] = get_dist_avg(avg_dists, i)
              
                avg_lock.release()


                
                
    except:
        avgs[0] = 'Error'
        avgs[4] =  str(sys.exc_info())

    return

def maintain_qs(out_q, buf_num ,num_cams, errors, occupants, avg_dists, e_lock, o_lock, d_lock):
    e_lock.acquire()
    o_lock.acquire()
    d_lock.acquire()
    errors = errors.append( [[None]* buf_num for _ in range(num_cams)])
    occupants = occupants.append([[None]* buf_num for _ in range(num_cams)])
    avg_dists = avg_dists.append([[None]* buf_num for _ in range(num_cams)])
    e_lock.release()
    o_lock.release()
    d_lock.release()

    index = 0
    rollover = 0
    
    #not sure that a is cycling through the buffers in the best way
    try:
        while(True):
            #location of oldest entry in mega list
            index = index % buf_num
            if not out_q.empty():
                # total = total+1
                rollover = rollover + 1
                data = out_q.get()
                #camera number
                i = data[0]
               
                #updates camera buffer at oldest index
                with e_lock:
                    errors[i][index] = data[1]
                with o_lock:
                    occupants[i][index] = data[2]
                with d_lock:
                    avg_dists[i][index] = data[3]
                
                if rollover == (num_cams):
                    rollover = 0
                    index = index + 1
    except:
        print('Unexpected error: ' + sys.exc_info())


def get_e_avg(errors, i, e_lock):
    with e_lock:
        error_avg = math.ceil(calc_avg(errors[i]))
        return error_avg
    
def get_o_avg(occupants, i, o_lock):
    with o_lock:
        occupant_avg = math.ceil(calc_avg(occupants[i]))
        return occupant_avg

def get_dist_avg(avg_dists, i, d_lock):
    with d_lock:
        dist_avg = round(calc_avg(avg_dists[i]), 2)
        return dist_avg





def calc_avg(num_list):
    total = 0
    count = 0
    for num in num_list:
        if num is not None:
            total = total + num
            count = count + 1
    if count != 0:
        avg = total/count
    else:
        avg = 0
        
    return avg    