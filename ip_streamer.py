# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:11:48 2020

@author: Nikki
"""

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
    ips = ['/home/worklab/Data/cv/video/AOTsample1_1.mp4']
    m = mp.Manager()
    updated = m.Value(c_bool, False)
    frames = m.list([None] * len(ips))
    times = m.list([None] * len(ips))

    #lock = manager.Lock()
    #start_lock = manager.Lock()
    streamers = []
    try:
        for i, ip in enumerate(ips):
            streamers.append(mp.Process(target=stream_all, args=(frames, times, ip, updated, i)))
        for streamer in streamers:
            streamer.start()
            
        while updated.value == False:
            continue
        
        while True:
            for i in range(len(ips)):
                #cv2.namedWindow("result" + str(i), cv2.WINDOW_NORMAL)
                cv2.imshow("result" + str(i), frames[i])
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except:
        print('Unexpected error: ', sys.exc_info())
        for streamer in streamers:
            streamer.terminate()
        cv2.destroyAllWindows()


def stream_all(frames, times, ip, updated, i):
    #list of ip addresses to get video from
    stream = open_cap(ip)
    # frames = [None] * len(ips)
    # times = [None] * len(ips)
    print(frames[i])
    print(times[i])
    print(stream)
    get_cap(stream, frames, times, i)

    updated.value = True
    try:
        while(True):
            get_cap(stream, frames, times, i)
            updated.value = True
           
        
            # cv2.namedWindow("result" + str(i), cv2.WINDOW_NORMAL)
            # cv2.imshow("result" + str(i), frames[i])
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
    except:
        print("Unexpected error:", sys.exc_info()[0])
        close_cap(stream)
        cv2.destroyAllWindows()
    return
    
#opens video capture objects for all input streams  
def open_cap(ip):
    print("Video from: ", ip)
    stream = cv2.VideoCapture(ip)
    print ("Capture opened")
    return stream

#gets the next frame from each video capture object
#at some point should read this directly into shared memory
def get_cap(stream, frames, times, i):
    times[i] = datetime.datetime.now()
    ret_val, frames[i] = stream.read()

#closes all video capture objects
def close_cap(stream):
    stream.release()
    print ("Captures closed")


if __name__ == "__main__":
    main()