# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:06:14 2020

@author: Nikki
"""
import cv2
import datetime
import sys
import addresses
import subprocess
import ctypes
import time

# ip_address = addresses.TEST
# ip_address2 = addresses.TEST2
ips = addresses.TEST #[addresses.TEST, addresses.TEST2]
RECORD = False
SHOW_VID = True
OUTPUT_VID = ['C:/Users/Nikki/Documents/work/inputs-outputs/vid_output/stream1.avi',
              'C:/Users/Nikki/Documents/work/inputs-outputs/vid_output/stream2.avi']


vid = [None]*len(ips)
for i, ip in enumerate(ips):
    print("Video from: ", ip)
    vid[i] = cv2.VideoCapture(ip)
    if RECORD:
        fps = vid[i].get(5)
        wdt = int(vid[i].get(3))
        hgt = int(vid[i].get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_vid[i] = cv2.VideoWriter(OUTPUT_VID[i], fourcc, fps, (wdt, hgt))

try:

        
    while True:
        for i, ip in enumerate(ips):    
            #skip desired number of frames to speed up processing
            #vid.grab()
            
            #get current time and next frame
            dt = str(datetime.datetime.now())
            #for calculating how long it takes to process a frame
            prev_time = time.time()
            
            return_value, frame = vid[i].read()
            curr_time = time.time()
            exec_time = curr_time - prev_time
            info = "time1: %.2f ms" %(1000*exec_time)
            print(info)
            # check that the next frame exists, if not, close display window and exit loop
            # if return_value:
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     #image = Image.fromarray(frame)
            # else:
            #     if SHOW_VID:
            #         cv2.destroyWindow('result')
            #     print('Video has ended')
            #     break
            if SHOW_VID:
                cv2.namedWindow("result" + str(i), cv2.WINDOW_NORMAL)
                cv2.imshow("result" + str(i), frame)
            if RECORD:
                 out_vid[i].write(frame)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        
        
    #end video, close viewer, stop writing to file
    for i in range(len(ips)):
        vid[i].release()
        print('Video has ended')
        if RECORD:
             out_vid[i].release()
        if SHOW_VID:
            cv2.destroyAllWindows()
    
#if interrupted, end video, close viewer, stop writing to file
except:
    print("Unexpected error:", sys.exc_info()[0])
    for i in range(len(ips)):
        print(i)
        vid[i].release()
        if RECORD == True:
             out_vid[i].release()
        if SHOW_VID:
            cv2.destroyAllWindows()
