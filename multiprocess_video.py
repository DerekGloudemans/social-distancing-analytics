# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:25:10 2020

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





def main():
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
    #length of queues, kinda arbitrary - this is the number that will be used for moving avg analytics
    buf_num = 3
    
    
    
    #  create manager to handle shared variables across processes
    updated = manager.Value(c_bool, False)
    frames = manager.list([None]* num_cams)
    times = manager.list([None]* num_cams)
    avgs = manager.list([None] * 5)
    avg_lock = manager.Lock()
    out_q = manager.Queue(num_cams*2)
    
    errs = manager.list()
    ocpts = manager.list()
    # for _ in range(num_cams):
    #     ocpts.append( [None]* buf_num)
    dists = manager.list()
    e_lock = manager.Lock()
    
    for i in range(num_cams):
        errs.append(manager.list([None]))
        ocpts.append(manager.list([None]))
        dists.append(manager.list([None]))


    #stores frame data that has been transfered to GPU
    gpu_frames = [None]* num_cams     
    
    workers = setup_gpus(gpu_list=[0])
    print(workers)
    
    
    #start model
    # model = detector.start_model()
    streamers = []

    try:
        #grab video frames in separate process
        for i, ip in enumerate(ips):
            streamer = streamers.append(mp.Process(target=ip_streamer.stream_all, args=(frames, times, ip, updated, i)))
        for streamer in streamers:
            streamer.start()
            print('Streamer process started')
        
        # analysis = mp.Process(target=adat.main, args=(all_output_stats, buf_num, avgs, removed))
        analysis = mp.Process(target=adat.main, args=(out_q, buf_num, num_cams, avgs, avg_lock, errs, ocpts, dists)) 
        analysis.start()
        print('Analysis process started')
        
        #wait until frames are starting to be read
        while(not updated.value):   
            continue
        
        #find and assign the frame size of each stream
        #TODO may have to resize frames or something for running through the model in batches
        print('here')
        for i, vid in enumerate(vids):
            vid.set_size(frames[i].shape[:2])
        print('here2')    
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
                    gpu_frames[i] = detector.frame_to_gpu(frame, workers[0].gpu)
            
                #loop through frames, find people, record occupants and infractions  
                for i, vid in enumerate(vids): 
                    frame = frames[i]
                    gpu_frame = gpu_frames[i]
                    dt = times[i]
                    bboxes = detector.person_bboxes(workers[0].model, gpu_frame, vid.frame_size)
                    
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
                    if out_q.full():
                        a = out_q.get()
                        print("Leaving queue: ", a)
                    out_q.put(stats)
                                       
                    # avg_lock.acquire()
                    # print("Avgs: ", avgs)
                    # avg_lock.release()
                    print(adat.get_o_avg(ocpts, i))
                    print(adat.get_e_avg(errs, i))
                    # print(avgs[0])
                    print(adat.get_dist_avg(dists, i))
                    # print(str(ocpts[i]))
                    # if len(ocpts) > 1:
                    #     print(adat.get_o_avg(ocpts, i))
                    
                    #save frames
                    if vid.frame_save:
                        outpt_frame(ftpts, frame, vid, errors, occupants, bboxes)
                        
                    # FIXME - just for debugging, show frame on screen
                    show_frame(ftpts, frame, vid, errors, occupants, bboxes, i)  
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
           
    except:
        print("Unexpected error:", sys.exc_info())
        cv2.destroyAllWindows()
        for vid in vids:
            vid.base_f.close()
        for streamer in streamers:
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

def setup_gpus(num = 0,  gpu_list = []):
    workers = []
   
    #load gpus from a list of numbers
    for a in gpu_list:
        workers.append(worker(a))
        
    #or load gpus starting at zero to a number
    for i in range(num):
        workers.append(worker(i))
        
    return workers
    #set up a model on each gpu
    #return a list of gpu workers
###---------------------------------------------------------------------------
#  

class worker():
    def __init__(self,i):
        #whether gpu is available
        self.avail = True
        
        #what gpu this worker is on
        self.gpu = "/gpu:" + str(i)
        
        #sets up a model on this gpu to predict with
        self.model = detector.start_model(self.gpu)
    
    def mark_avail(self):
        self.avail = True
        
    def mark_unavail(self):
        self.avail = False
        
    def set_frame(self, frame):
        self.cur_frame = frame
        
        

               
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