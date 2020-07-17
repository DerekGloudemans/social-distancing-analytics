# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:11:48 2020

@author: Nikki
"""
import addresses as ad
import cv2
import datetime
import sys
import multiprocessing as mp
import time
from ctypes import c_bool

#ip streams
#multiple video cap objsects

#change detection to output to csv, save every 5 min or so

#make separate processes
#need to check that frame grabbed exists and hasn't already been processed
def main():
    ips = ad.AOT
    frames = [None] * len(ips)
    times = [None] * len(ips)
    m = mp.Manager()
    updated = m.Value(c_bool, False)
    #lock = manager.Lock()
    #start_lock = manager.Lock()
    
    stream_all(frames, times, ips,updated)

def stream_all(frames, times, ips, updated):
    #list of ip addresses to get video from
    streams = open_cap(ips)
    # frames = [None] * len(ips)
    # times = [None] * len(ips)
    print('here')
    frames, times = get_cap(streams, frames, times)
    updated.value = True
    try:
        while(True):
            # prev_time = time.time()

            #print('Check 1: ', times[0])
            
            frames, times = get_cap(streams, frames, times)
            updated.value = True
            # frames = [0,0,3]
            # times = [0,1,2]
            # frames = a
            # times = b
            

            # c_frames.send(frames)
            # c_times.send(times)
            # for i, frame in enumerate(frames):
            #     cv2.namedWindow("result" + str(i), cv2.WINDOW_NORMAL)
            #     cv2.imshow("result" + str(i), frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            # curr_time = time.time()
            # exec_time = curr_time - prev_time
            # info = "time: %.2f ms" %(1000*exec_time/len(ips))
            # print(info)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        close_cap(streams)
        cv2.destroyAllWindows()
    return
    
#opens video capture objects for all input streams  
def open_cap(ips):
    streams = [None] * len(ips)
    for i, ip in enumerate(ips):
        print("Video from: ", ip)
        streams[i] = cv2.VideoCapture(ip)
    print ("Captures opened")
    return streams

#gets the next frame from each video capture object
#at some point should read this directly into shared memory
def get_cap(streams, frames, times):
    for i, stream in enumerate(streams):
        times[i] = datetime.datetime.now()
        ret_val, frames[i] = stream.read()
    #print('Check 2: ', times[0])    
    return frames, times

#closes all video capture objects
def close_cap(streams):
    for i, stream in enumerate(streams):
        stream.release()
    print ("Captures closed")

def sample(num):
    print(num)
    return

if __name__ == "__main__":
    main()