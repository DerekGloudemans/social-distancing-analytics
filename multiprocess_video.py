# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:25:10 2020

@author: Nikki
"""

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
import pixel_gps as pg
import detector
#uncomment to verify that GPU is being used
#tf.debugging.set_log_device_placement(True)


def main():
    try:
        print(mp.cpu_count())
        ips = addresses.AOT
        #print(ips)
        buf_len = len(ips)
        # variable manager
        manager = mp.Manager()
        frames = manager.list([None]* buf_len)
        times = manager.list([None]* buf_len)
        #start_lock = manager.Lock()
        #lock = manager.Lock()
        updated = mp.Value(c_bool, False)
        # p_frames, c_frames = mp.Pipe()
        # p_times, c_times = mp.Pipe()
    
        all_vid_info = detector.get_info(['aot1', 'aot2'])
        model = detector.start_model()
        files = [None]* buf_len
        
        for i in range(buf_len):
            output_f = 'C:/Users/Nikki/Documents/work/inputs-outputs/txt_output/mp_test' + str(i) +'.txt'
            files[i] = open(output_f, 'w')
            print('file started')
            files[i].write('Time\t\t\t\tPed\t<6ft\n')
        
        streamer = mp.Process(target=ip_streamer.stream_all, args=(frames, times, ips, updated)) #args=(frames, times, ips))
        print('yes')
        streamer.start()
        while(True):
            if updated.value == True:

                prev_time = time.time()
                # if p_frames.poll() and p_times.poll():
                #print('received')
                # frames = p_frames.recv()
                # times = p_times.recv()
                
                
                for i, frame in enumerate(frames):
                    result = detector.find_occupants(frame, times[i], model, all_vid_info[i], files[i])
                    
                    # print(times[i])
                    cv2.namedWindow("result" + str(i), cv2.WINDOW_NORMAL)
                    cv2.imshow("result" + str(i), result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                updated.value = False
                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = "time: %.2f ms" %(1000*exec_time/buf_len)
                print(info)
    except:
        print("Unexpected error:", sys.exc_info())
        cv2.destroyAllWindows()
        for f in files:
            f.close()
        streamer.terminate()
        

            
if __name__ == '__main__':
    main()