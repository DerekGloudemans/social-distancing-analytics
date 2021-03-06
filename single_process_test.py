# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:25:10 2020

@author: Nikki, Derek Gloudemans
"""

import torch
import torchvision.transforms.functional as F
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

import pixel_realworld as pr
import transform as tform
import csv
import ast
import analyze_data as adat
import utils 

# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"models","retinanet")
sys.path.insert(0,detector_path)
from models.retinanet.retinanet import model




def main(errs, ocpts, dists, updated, frames, times, avgs, avg_lock, i_lock, ind,out_q, bbox_q, image_q, config,ctx):
    
    # file containing camera information
    # transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/transforms.csv'
    # transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/test.csv'
    # transform_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/test_all.csv'
    # transform_f = './config/LAMBDA_TEST.config'
    
    
    
    #create VidObjs to store information about each camera
    cameras = initialize_cams(config)
    
    num_cams = len(cameras)
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

    classes = utils.read_class_names("./config/coco.names")

    #stores frame data that has been transfered to GPU
    GPU_LIST = [i for i in range(torch.cuda.device_count())]
    # workers = setup_gpus(vids, gpu_list = GPU_LIST)
    #start model
    # model = detector.start_model()
    streamers = []

    worker = Worker(2)
    frame = cv2.imread("/home/worklab/Desktop/test1.png")
    worker.set_frame(frame)
    ped,veh = worker.get_bboxes()
    cameras[0]["frame_size"] = (worker.gpu_frame[0].shape[:2])

    im = F.normalize(worker.gpu_frame[0],mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
    im = F.to_pil_image(im.cpu())
    open_cv_image = np.array(im)
    im = open_cv_image.copy()/255.0
    im = im[:,:,::-1]
    #combine so bounding boxes remain associated with camera
    i = 0
    box_ind = (ped,veh,i,im)
    
    ped_bboxes = box_ind[0]
    veh_bboxes = box_ind[1]
    i = box_ind[2]
    frame = box_ind[3]
    
    # first, try and show frame
    #frame = frame.transpose(1, 2, 0)
    # cv2.imshow("test",frame)
    # cv2.waitKey(0)
    
    camera = cameras[i]
    filename = camera["address"]
    pix_real = camera["im-gps"]
    frame_show = camera["save_frames"]
    dt = times[i]
    
    #find ft pts and convert to real_world
    ped_pts = utils.get_ftpts(ped_bboxes)
    realpts = tform.transform_pt_array(ped_pts, pix_real)
    
    # verifies there is more than one point in the list (each point has size 2)]
    if realpts.size > 2:
        mytree = scipy.spatial.cKDTree(realpts)
        errors = utils.compliance_count(mytree, realpts)
        
        #FIXME can probably do these both in 1 function
        avg_dist = utils.find_dist(mytree, realpts)
        avg_min_dist = utils.find_min_dist(mytree, realpts)
    else:
        errors = 0
        avg_min_dist = None
        avg_dist = None
    occupants = len(ped_bboxes)
    #output info to csv file  
    with open(filename, 'a', newline='') as base_f:
        writer = csv.writer(base_f)
        utils.video_write_info(writer, realpts, str(dt), errors, occupants, avg_dist, avg_min_dist)
            
    stats = [i, errors, occupants, avg_min_dist]
    
    #put outpt data into queue so it is accessible by the analyzer
    if out_q.full():
        out_q.get()
    out_q.put(stats)
    
    
    result = prep_frame(ped_pts, frame, camera, errors, occupants, ped_bboxes,veh_bboxes,classes)
    
    cv2.imshow("frame",result)
    cv2.waitKey(0)

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
 
def prep_frame(ftpts, frame, camera, errors, occupants, ped_bboxes,veh_bboxes,classes):

    pix_real = camera["im-gps"]
    real_pix = camera["gps-im"]
    origin = camera["estimated_camera_location"] 

    frame_size = camera["frame_size"]
    print(frame_size)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(ped_bboxes) > 0:
        frame = utils.draw_bbox(frame, ped_bboxes, classes,show_label = True,redact = True)
        frame, x = pr.draw_radius(frame, ftpts, real_pix, pix_real, origin)
    utils.overlay_occupancy(frame, errors, occupants, frame_size)
    
    if len(veh_bboxes) > 0:
        frame = utils.draw_bbox(frame, veh_bboxes, classes,show_label = False,redact = False)
    
    #result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


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
             
                 if pval == "None":
                    pval = None
                 if pval == "True":
                    pval = True
                 if pval == "False":
                    pval = False
                    
             current_block[pkey] = pval
             
         else:
             raise AttributeError("""Got a line in the configuration file that isn't a block header nor a 
             key=value.\nLine: {}""".format(strip_line))
     
 return all_blocks

def initialize_cams(transform_f):
    
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

class Worker():
    # def __init__(self,i,vids):
    def __init__(self, i):
        #whether gpu is available
        self.avail = True
        # self.vids = vids
        
        self.conf_cutoff = 0.3
        
        #what gpu this worker is on
        try:
            self.gpu = torch.device(i)
        except:
            self.gpu = torch.device("cpu")
        
        torch.cuda.set_device(i)
        
        #sets up a model on this gpu to predict with
        state_dict_path = "./config/coco_resnet_50_map_0_335_state_dict.pt"
        self.model = model.resnet50(num_classes = 80)
        self.model.load_state_dict(torch.load(state_dict_path))
        self.model = self.model.to(self.gpu)
        self.model.eval()
        self.model.training = False  
        self.model.freeze_bn()
        
        
    def mark_avail(self):
        self.avail = True
        
    def mark_unavail(self):
        self.avail = False
        
    def set_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = F.to_tensor(frame)
        frame = F.normalize(frame,mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        frame = frame.unsqueeze(0)
        self.gpu_frame = frame.to(self.gpu)
        
        
    def get_bboxes(self):
        confs,classes,bboxes = self.model(self.gpu_frame)
        
        # keep only high confidence objects
        obj_idxs = torch.where(confs > self.conf_cutoff)[0]
        confs = confs[obj_idxs]
        classes = classes[obj_idxs]
        bboxes = bboxes[obj_idxs]
        
        # 0: 'person'
        # 1: 'bicycle'
        # 2: 'car'
        # 3: 'motorcycle'
        # 4: 'airplane'
        # 5: 'bus'
        # 6: 'train'
        # 7: 'truck'
        ped_idxs = torch.cat([torch.where(classes == 0)[0],torch.where(classes == 1)[0]],dim = 0)
        veh_idxs = torch.cat([torch.where(classes == 5)[0],torch.where(classes == 2)[0],torch.where(classes == 3)[0],torch.where(classes == 7)[0]],dim = 0)
        
        out = torch.cat([bboxes,classes.float().unsqueeze(1),confs.unsqueeze(1)],dim = 1)
        
        peds = out[ped_idxs].data.cpu()
        vehs = out[veh_idxs].data.cpu()
        
        return peds,vehs
         
        # parse out pedestrians into one group, vehicles into another group
        
    #given a shared variable i, loops constantly and processes frame at i
    #i should probably have a lock      

# def proc_video(worker, ind, i_lock, frames, times, out_q):
def proc_video(ind, i_lock, frames, times, bbox_q, cameras, gpu):
        
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
                    camera = cameras[i]
                #TODO could benefit from a lock, would help ensure frame and time are actually corresponding
                #pretty sure manager objects already have lock control so the smae item isn't accessed from separate processes at once
                #but that could also be a good lock to have
                worker.set_frame(np.asarray(frames[i]))
                ped_bboxes,veh_bboxes = worker.get_bboxes()
                
                # denormalize
                im = F.normalize(worker.gpu_frame[0],mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
                im = F.to_pil_image(im.cpu())
                open_cv_image = np.array(im)
                im = open_cv_image.copy()/255.0
                im = im[:,:,::-1]
                #combine so bounding boxes remain associated with camera
                box_ind = (ped_bboxes,veh_bboxes,i,im)
                
                bbox_q.put(box_ind)
                #bboxes should be sent to a queue, should also have frame or camera number associated
                
                worker.mark_avail()
                
    except KeyboardInterrupt:
        print("Unexpected worker error:", sys.exc_info()[0])
    return
               
###---------------------------------------------------------------------------
#   input queue of bbox info, output queue of stats


#reassociate output boxes with frame or at least video number
#give it a queueue of bboxes and associated, (frame/video number)
#willl be a separate process
#monitor queue size so it doesn't get ridiciulously big

#could move writing to a different process but probably not atm
def post_processor(bbox_q, cameras, out_q, frames, times, image_q = None):
    classes = utils.read_class_names("./config/coco.names")

    try:
        while True:
            if not bbox_q.empty():
                box_ind = bbox_q.get()
                ped_bboxes = box_ind[0]
                veh_bboxes = box_ind[1]
                i = box_ind[2]
                frame = box_ind[3]
                
                # first, try and show frame
                #frame = frame.transpose(1, 2, 0)
                # cv2.imshow("test",frame)
                # cv2.waitKey(0)
                
                camera = cameras[i]
                filename = camera["address"]
                pix_real = camera["im-gps"]
                frame_show = camera["save_frames"]
                dt = times[i]
                
                #find ft pts and convert to real_world
                ped_pts = utils.get_ftpts(ped_bboxes)
                realpts = tform.transform_pt_array(ped_pts, pix_real)
                
                # verifies there is more than one point in the list (each point has size 2)]
                if realpts.size > 2:
                    mytree = scipy.spatial.cKDTree(realpts)
                    errors = utils.compliance_count(mytree, realpts)
                    
                    #FIXME can probably do these both in 1 function
                    avg_dist = utils.find_dist(mytree, realpts)
                    avg_min_dist = utils.find_min_dist(mytree, realpts)
                else:
                    errors = 0
                    avg_min_dist = None
                    avg_dist = None
                occupants = len(ped_bboxes)
                #output info to csv file  
                with open(filename, 'a', newline='') as base_f:
                    writer = csv.writer(base_f)
                    utils.video_write_info(writer, realpts, str(dt), errors, occupants, avg_dist, avg_min_dist)
                        
                stats = [i, errors, occupants, avg_min_dist]
                
                #put outpt data into queue so it is accessible by the analyzer
                if out_q.full():
                    out_q.get()
                out_q.put(stats)
                
                
                result = prep_frame(ped_pts, frame, camera, errors, occupants, ped_bboxes,veh_bboxes,classes)
                
                cv2.imshow("frame",result)
                cv2.waitKey(0)
                
                # if frame_save or frame_show:
                #     result = prep_frame(ftpts, frame, vid, errors, occupants, bboxes)
                
                ### UNCOMMENT AND USE
                # frame_save = True
                # #save frames
                # if frame_save:
                #     outpt_frame(result, vid)
                    
                # if frame_show:
                #     if image_q.full():
                #         image_q.get()
                #     image_q.put(result)
                    
                # # FIXME - just for debugging, show frame on screen
                show_frame(result, i)  
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        print("Unexpected postprocessing error:", sys.exc_info()[0])
        cv2.destroyAllWindows()
    return
###---------------------------------------------------------------------------
#   
# def fill_vid_array(name, pix_real, real_pix, save = False):
#     #make an array that's defined # wide, undef tall
#     #fill corresponding # of fields
#     #row # = camera num
#     #TODO could ake this a dictionary to make it easier to understand
#     #0) filename, 1) frame_save, 2) pix_real, 3) real_pix, 4) origin, 5) frame_dir, 6) count, 7) frame_size
#     # filename = vid[0]
#     # frame_save = vid[1]
#     # pix_real = vid[2]
#     # real_pix = vid[3]
#     # origin = vid[4] 
#     # frame_dir = vid[5]
#     # count = vid[6]
#     # frame_size = vid[7]
    
#     #name of output file
#     filename = 'C:/Users/Nikki/Documents/work/inputs-outputs/txt_output/' + name + '.csv'
    
#     #create base file and csv writer to add to file
#     # base_f = open(filename, 'a', newline='')
#     # csvfile = csv.writer(base_f)
    
#     origin = np.array([0,0])
        
#     #set output directory and frame number in case video is to be saved
#     frame_dir = 'C:/Users/Nikki/Documents/work/inputs-outputs/vid_output/' + name + '_frames/'
#     count = 0  
    
#     frame_size = 0 #will get updated later        
#     print('Saving frames: ', save)
    
#     cur_vid = np.array([filename, save, pix_real, real_pix, origin, frame_dir, count, frame_size])
#     return cur_vid
            

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
    ctx = mp.get_context('spawn')
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
    num_cams = 1

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
        
    main(errs, ocpts, dists, updated, frames, times, avgs, avg_lock, i_lock, ind,out_q, bbox_q, image_q,config,ctx)
        
    # out = cv2.imread("test.jpg")
    # test = Worker(1)
    # test.set_frame(out)
    # test.get_bboxes()