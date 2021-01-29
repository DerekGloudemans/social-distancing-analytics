# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:25:10 2020

@author: Nikki
"""

# numpy, scipy, tensorflow
#tensorflow + weights, flask

import importlib
import multiprocessing as mp
import ip_streamer
import cv2
import sys
import numpy as np
import time
from ctypes import c_bool
import scipy.spatial
import os

import tensorflow as tf
from core import utils
import pixel_realworld as pr
import detector
import transform as tform
import csv
import ast
import analyze_data as adat





def main(errs, ocpts, dists, updated, frames, times, avgs, avg_lock, i_lock, ind,out_q, bbox_q, image_q, config):
# def main():
    #uncomment to verify that GPU is being used
    tf.debugging.set_log_device_placement(False)
    # importlib.reload(mp)
    ips = []
    vids = np.array([[]])
    
    # file containing camera information
    # transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/transforms.csv'
    # transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/test.csv'
    # transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/test_all.csv'
    # transform_f = './config/LAMBDA_TEST.config'
    
    
    
    #create VidObjs to store information about each camera
    vids = initialize_cams(config, ips, vids)
    
    num_cams = len(vids)
    #length of queues, kinda arbitrary - this is the number that will be used for moving avg analytics
    buf_num = 3
    
    
    #need to fix these references
    # errs = var_list[0]
    # ocpts = var_list[1]
    # dists = var_list[2]
    # updated = var_list[3]
    # frames = var_list[4]
    # times = var_list[5]
    # avgs = var_list[6]
    # avg_lock = var_list[7]
    # i_lock = var_list[8]
    # ind = var_list[9]
    # bbox_q = var_list[10]
    # ind = var_list[11]
    
    #uncomment if running from this file
    
    # manager = mp.Manager()
    # print('MP manager created')
    # #  create manager to handle shared variables across processes
    # updated = manager.Value(c_bool, False)
    # frames = manager.list([None]* num_cams)
    # times = manager.list([None]* num_cams)
    # avgs = manager.list([None] * 5)
    # avg_lock = manager.Lock()
    # i_lock = manager.Lock()
    # out_q = manager.Queue(num_cams*2)
    # bbox_q = manager.Queue()
    # ind = manager.Value(int, 0)
    # image_q = manager.Queue(num_cams*2)
    
    # # sample = manager.list([None] * 5)

    # errs = manager.list()
    # ocpts = manager.list()
    # dists = manager.list()
    # s_lock = manager.Lock()
    # # for _ in range(num_cams):
    # #     ocpts.append( [None]* buf_num)
    
    # for i in range(num_cams):
    #     errs.append(manager.list([None]))
    #     ocpts.append(manager.list([None]))
    #     dists.append(manager.list([None]))


    #stores frame data that has been transfered to GPU
    GPU_LIST = []#[0]
    # workers = setup_gpus(vids, gpu_list = GPU_LIST)
    #start model
    # model = detector.start_model()
    streamers = []

    try:
        #grab video frames in separate process
        for i, ip in enumerate(ips):
            streamer = mp.Process(target=ip_streamer.stream_all, args=(frames, times, ip, updated, i))
            streamers.append(streamer)
        for streamer in streamers:
            streamer.daemon = True
            
            streamer.start()
            print('Streamer processes started')
        # analysis = mp.Process(target=adat.main, args=(all_output_stats, buf_num, avgs, removed))
        analysis = mp.Process(target=adat.main, args=(out_q, buf_num, num_cams, avgs, avg_lock, errs, ocpts, dists)) 
        analysis.daemon = True
        analysis.start()
        print('Analysis process started')
        
        
            
        errs[0][0] = 4
        #wait until frames are starting to be read
        while(not updated.value):   
            continue
        errs[0][0] = 5
        time.sleep(2)
        #find and assign the frame size of each stream
        #TODO may have to resize frames or something for running through the model in batches
        for i, vid in enumerate(vids):
            #TODO coould make a set_size function for easier use
            vid[7] = (frames[i].shape[:2])
        prev_time = time.time()
        
        work_processes = []
        logic_gpus = False
        #need better gpu setup probably
        if logic_gpus:
            p_devices = tf.config.list_physical_devices('GPU')
            gpu = p_devices[0]
            #gtx 770 has 2048 mb total
            tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=100),
                                                            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100),
                                                            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])
            logical_devices = tf.config.list_logical_devices('GPU')
            print("Simulated GPUS: ", len(logical_devices))
            # print(logical_devices)
            # print(type(logical_devices[0]))
            
            
            for gpu in logical_devices:
                work_processes.append(mp.Process(target=proc_video, args=(ind, i_lock,frames, times, bbox_q, vids, gpu)))
        else: 
            if len(GPU_LIST) > 0:
                for gpu in GPU_LIST:
                    work_processes.append(mp.Process(target=proc_video, args=(ind, i_lock,frames,times,bbox_q, vids, gpu)))
            else:
                work_processes.append(mp.Process(target=proc_video, args=(ind, i_lock,frames,times,bbox_q, vids, gpu)))
        for proc in work_processes:
            proc.daemon = True
            proc.start()
        print('Worker processes started') 
        
        post_proc = mp.Process(target=post_processor, args=(bbox_q, vids, out_q, frames, times, image_q))
        post_proc.daemon = True
        post_proc.start()
        print('Post process started')    
        #make this main process be responsible for saving at intervals
        
        #itereate through frames with a shared ind 
        
        
        #make a function that does this
        #make a process for every worker
        #every wr=orker will constantly eb running this
        #needs an overarching queue or pointer o correct location in list to process
        
        #continuously loop until keyboard interrupt
        
        while(True):
            continue
            # with s_lock:
            #     print(sample)
    #         curr_time = time.time()
            
    #         # save outputs every 5 minutes
    #         if (curr_time - prev_time) > (5 * 60):
    #             save_files(vids)
    #             prev_time = curr_time
            
  
    except:
        print("Unexpected error:", sys.exc_info())
        cv2.destroyAllWindows()
            
        for streamer in streamers:
            streamer.terminate()
            
        analysis.terminate()
        
        for proc in work_processes:
            proc.terminate()
            
        post_proc.terminate()
    return

###---------------------------------------------------------------------------
#   Saves frame with overlaid info
          
def outpt_frame(result, vid):

    frame_dir = vid[5]
    count = vid[6]
    

    cv2.imwrite(frame_dir + str(count) + '.jpg', result)
    count = count + 1
  
    
###---------------------------------------------------------------------------
#   Shows frame with overlaid info
          
def show_frame(result, i):

    cv2.namedWindow("result" + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("result" + str(i), result) 
    
        

###---------------------------------------------------------------------------
#   Overlays ellipses, bboxes, stats on frame
 
def prep_frame(ftpts, frame, vid, errors, occupants, bboxes):

    pix_real = vid[2]
    real_pix = vid[3]
    origin = vid[4] 

    frame_size = vid[7]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(bboxes) > 0:
        frame = utils.draw_bbox(frame, bboxes, show_label = False)
        frame, x = pr.draw_radius(frame, ftpts, real_pix, pix_real, origin)
    utils.overlay_occupancy(frame, errors, occupants, frame_size)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return result



###---------------------------------------------------------------------------
#   Closes and reopens csvfiles to save them
   
def save_files(vids):
    # filename = vid[0]
    # frame_save = vid[1]
    # pix_real = vid[2]
    # real_pix = vid[3]
    # origin = vid[4] 
    # frame_dir = vid[5]
    # count = vid[6]
    # frame_size = vid[7]
    for vid in vids:
        vid.base_f.close()
        vid.base_f = open(vid.filename, 'a', newline='')
        vid.csvfile = csv.writer(vid.base_f)
        print('Saved')


###---------------------------------------------------------------------------
#   Creates VidObjs from addresses listed in csvfile
def parse_config_file(config_file):
 all_blocks = []
 current_block = None
 with open(config_file, 'r') as f:
     for line in f:
         # ignore empty lines and comment lines
         if line is None or len(line.strip()) == 0 or line[0] == '#':
             
             if current_block is not None:
                 # stack im and gps points into np arrays
                 current_block['impts'] = np.stack([current_block['im1'],current_block['im2'],current_block['im3'],current_block['im4']])
                 del current_block['im1']
                 del current_block['im2']
                 del current_block['im3']
                 del current_block['im4']
             
                 current_block['gpspts'] = np.stack([current_block['gps1'],current_block['gps2'],current_block['gps3'],current_block['gps4']])
                 del current_block['gps1']
                 del current_block['gps2']
                 del current_block['gps3']
                 del current_block['gps4']
                 
                 all_blocks.append(current_block)
                 current_block = None
                
             continue
         strip_line = line.strip()
         if len(strip_line) > 2 and strip_line[:2] == '__' and strip_line[-2:] == '__':
             # this is a configuration block line
             # first check if this is the first one or not
             if current_block is not None: 
                 all_blocks.append(current_block)
             current_block = {}
             current_block["name"] = str(strip_line[2:-2])

         elif '==' in strip_line:
             pkey, pval = strip_line.split('==')
             pkey = pkey.strip()
             pval = pval.strip().split("#")[0] # remove trailing comments
             
             # parse out coordinate values
             try:
                 pval1,pval2 = pval.split(",")
                 pval1 = float(pval1)
                 pval2 = float(pval2)
                 pval = np.array([pval1,pval2])
             except ValueError:
                 try:    
                     pval = float(pval)    
                 except ValueError:
                     pass 
                 
             current_block[pkey] = pval
             
         else:
             raise AttributeError("""Got a line in the configuration file that isn't a block header nor a 
             key=value.\nLine: {}""".format(strip_line))
     
 return all_blocks

def initialize_cams(transform_f, ips, vids):
    
    cameras = parse_config_file(config)
    keepers = [] 
    for camera in cameras:
        # verify address is valid
        test = cv2.VideoCapture(camera["address"])
        if test.isOpened():
            test.release()
            # get perspective transforms
            camera["im-gps"] = tform.get_best_transform(camera["impts"],camera["gpspts"])
            camera["gps-im"] = tform.get_best_transform(camera["gpspts"],camera["impts"])
            keepers.append(camera)
               
    return keepers
###---------------------------------------------------------------------------
#       

def setup_gpus(vids, num = 0,  gpu_list = []):
    
    workers = []
   
    #load gpus from a list of numbers
    for a in gpu_list:
        workers.append(Worker(a, vids))
        
    #or load gpus starting at zero to a number
    for i in range(num):
        workers.append(Worker(i, vids))
        
    return workers
    #set up a model on each gpu
    #return a list of gpu workers
###---------------------------------------------------------------------------
#  

class Worker():
    # def __init__(self,i,vids):
    def __init__(self, i):
        #whether gpu is available
        self.avail = True
        # self.vids = vids
        
        #what gpu this worker is on
        
        if isinstance(i, int):
            self.gpu = "/gpu:" + str(i)            
        elif isinstance(i, str):
            self.gpu = i
        elif i is None:
            self.gpu = None
        else:
            raise ValueError('The gpu is not properly setup, and the CPU is not working')
            
        #sets up a model on this gpu to predict with
        if self.gpu is None:
            self.model = detector.start_model()
    
    def mark_avail(self):
        self.avail = True
        
    def mark_unavail(self):
        self.avail = False
        
    def set_frame(self, frame):
        self.cur_frame = frame
        if self.gpu is None:
            self.gpu_frame = self.cur_frame
        else:
            self.gpu_frame = detector.frame_to_gpu(self.cur_frame, self.gpu)
            
        
    def get_bboxes(self, frame_size):
        return detector.person_bboxes(self.model, self.gpu_frame, frame_size)
        
    #given a shared variable i, loops constantly and processes frame at i
    #i should probably have a lock      

# def proc_video(worker, ind, i_lock, frames, times, out_q):
def proc_video(ind, i_lock, frames, times, bbox_q, vids, gpu):

    worker = Worker(gpu)
    try:
        while(True):
            if worker.avail:
                worker.mark_unavail()
                # save current index so it doesn't change while processing
                #lock ensures that multiple processes aren't working on same frame
                with i_lock:
                    i = ind.value
                    ind.value = ind.value + 1
                    ind.value = ind.value % len(frames)
                    #loop through frames, find people, record occupants and infractions 
                    #TODO not sure the best way to protect smae vid from being accessed simultaneously
                    vid = vids[i]
                frame_size = vid[7]
                #TODO could benefit from a lock, would help ensure frame and time are actually corresponding
                #pretty sure manager objects already have lock control so the smae item isn't accessed from separate processes at once
                #but that could also be a good lock to have
                worker.set_frame(np.asarray(frames[i]))
                bboxes = worker.get_bboxes(frame_size)
                
                #combine so bounding boxes remain associated with camera
                box_ind = np.array([bboxes, i, worker.cur_frame])
                
                bbox_q.put(box_ind)
                #bboxes should be sent to a queue, should also have frame or camera number associated
                
                
                
                worker.mark_avail()
    except:
        print("Unexpected error:", sys.exc_info()[0])
    return
               
###---------------------------------------------------------------------------
#   input queue of bbox info, output queue of stats


#reassociate output boxes with frame or at least video number
#give it a queueue of bboxes and associated, (frame/video number)
#willl be a separate process
#monitor queue size so it doesn't get ridiciulously big

#could move writing to a different process but probably not atm
def post_processor(bbox_q, vids, out_q, frames, times, image_q = None):
    try:
        while True:
            if not bbox_q.empty():
                box_ind = bbox_q.get()
                bboxes = box_ind[0]
                i = box_ind[1]
                frame = box_ind[2]
                
                vid = vids[i]
                filename = vid[0]
                frame_save = vid[1]
                pix_real = vid[2]
                
                dt = times[i]
                
                #find ft pts and convert to real_world
                ftpts = utils.get_ftpts(bboxes)
                realpts = tform.transform_pt_array(ftpts, pix_real)
                # verifies there is more than one p[oint in the list (each point has size 2)]
                if realpts.size > 2:
                    mytree = scipy.spatial.cKDTree(realpts)
                    errors = detector.compliance_count(mytree, realpts)
                    
                    #FIXME can probably do these both in 1 function
                    avg_dist = detector.find_dist(mytree, realpts)
                    avg_min_dist = detector.find_min_dist(mytree, realpts)
                else:
                    errors = 0
                    avg_min_dist = None
                    avg_dist = None
                occupants = ftpts.size//2
                #output info to csv file  
                with open(filename, 'a', newline='') as base_f:
                    writer = csv.writer(base_f)
                    utils.video_write_info(writer, realpts, str(dt), errors, occupants, avg_dist, avg_min_dist)
                        
                stats = [i, errors, occupants, avg_min_dist]
                
                #put outpt data into queue so it is accessible by the analyzer
                if out_q.full():
                    out_q.get()
                out_q.put(stats)
                
                #FIXME this should be set somewhere else
                frame_show = True
                
                result = prep_frame(ftpts, frame, vid, errors, occupants, bboxes)
                
                # if frame_save or frame_show:
                #     result = prep_frame(ftpts, frame, vid, errors, occupants, bboxes)

                #save frames
                if frame_save:
                    outpt_frame(result, vid)
                    
                if frame_show:
                    if image_q.full():
                        image_q.get()
                    image_q.put(result)
                    
                # # FIXME - just for debugging, show frame on screen
                # show_frame(result, i)  
                # if cv2.waitKey(1) & 0xFF == ord('q'): break
    except:
        print("Unexpected error:", sys.exc_info()[0])
        cv2.destroyAllWindows()
    return
###---------------------------------------------------------------------------
#   
def fill_vid_array(name, pix_real, real_pix, save = False):
    #make an array that's defined # wide, undef tall
    #fill corresponding # of fields
    #row # = camera num
    #TODO could ake this a dictionary to make it easier to understand
    #0) filename, 1) frame_save, 2) pix_real, 3) real_pix, 4) origin, 5) frame_dir, 6) count, 7) frame_size
    # filename = vid[0]
    # frame_save = vid[1]
    # pix_real = vid[2]
    # real_pix = vid[3]
    # origin = vid[4] 
    # frame_dir = vid[5]
    # count = vid[6]
    # frame_size = vid[7]
    
    #name of output file
    filename = 'C:/Users/Nikki/Documents/work/inputs-outputs/txt_output/' + name + '.csv'
    
    #create base file and csv writer to add to file
    # base_f = open(filename, 'a', newline='')
    # csvfile = csv.writer(base_f)
    
    origin = np.array([0,0])
        
    #set output directory and frame number in case video is to be saved
    frame_dir = 'C:/Users/Nikki/Documents/work/inputs-outputs/vid_output/' + name + '_frames/'
    count = 0  
    
    frame_size = 0 #will get updated later        
    print('Saving frames: ', save)
    
    cur_vid = np.array([filename, save, pix_real, real_pix, origin, frame_dir, count, frame_size])
    return cur_vid
            

#TODO make function to easily edit size, saving setting

# class VidObj:
    
#     def __init__(self, name, pix_real, real_pix, save = False):
#         #name of output file
#         self.filename = 'C:/Users/Nikki/Documents/work/inputs-outputs/txt_output/' + name + '.csv'
        
#         #create base file and csv writer to add to file
#         self.base_f = open(self.filename, 'a', newline='')
#         self.csvfile = csv.writer(self.base_f)
        
#         #set whether or not to save file
#         self.frame_save = save
        
#         #set transformation matrices
#         self.pix_real = pix_real
#         self.real_pix = real_pix
#         self.origin = np.array([0,0])
#         # self.transform_info = pr.sample_select(name)
        
#         #set output directory and frame number in case video is to be saved
#         self.frame_dir = 'C:/Users/Nikki/Documents/work/inputs-outputs/vid_output/' + name + '_frames/'
#         self.count = 0  
                
#         print('Saving frames: ', self.frame_save)
        
        
#     def set_size(self, frame_size):
#         self.frame_size = frame_size
#         self.origin = np.array([frame_size[1]/2, frame_size[0]])
        
#     def start_save(self):
#         self.frame_save = True
        
#     def end_save(self):
#         self.frame_save = False
        
            
if __name__ == '__main__':
    manager = mp.Manager()
    buf_num = 3
    global errs
    global ocpts
    global dists
    errs = manager.list()
    ocpts = manager.list()
    dists = manager.list()
    config = './config/LAMBDA_TEST.config'

    #FIXME need a better way to do this (should be based on how many cameras initialize)
    #should initialize cameras here instead of in mp vid
    num_cams = 2

    updated = manager.Value(c_bool, False)
    frames = manager.list([None]* num_cams)
    times = manager.list([None]* num_cams)
    avgs = manager.list([None] * 5)
    avg_lock = manager.Lock()
    i_lock = manager.Lock()
    out_q = manager.Queue(num_cams*2)
    bbox_q = manager.Queue()
    ind = manager.Value(int, 0)
    
    global image_q
    image_q = manager.Queue(num_cams*2)
    
    for i in range(num_cams):
        errs.append(manager.list([None]))
        ocpts.append(manager.list([None]))
        dists.append(manager.list([None]))
        
    main(errs, ocpts, dists, updated, frames, times, avgs, avg_lock, i_lock, ind,out_q, bbox_q, image_q,config)