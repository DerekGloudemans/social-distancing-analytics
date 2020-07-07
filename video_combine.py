# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:20:58 2020

@author: Nikki
"""
import cv2
import sys
import numpy as np

#combines 2 videos into 1
OUTPUT_VID = './data/vid_output/aot_combo.avi'
video_path1 = './data/vid_output/aot1.mp4'
video_path2 = './data/vid_output/aot2.mp4'

#open videos
print("Video from: ", video_path1, ", ", video_path2)
vid2 = cv2.VideoCapture(video_path1)
vid1 = cv2.VideoCapture(video_path2)

#start writing
fps1 = vid1.get(5)
wdt1 = int(vid1.get(3))
hgt1 = int(vid1.get(4))

fps2 = vid2.get(5)
wdt2 = int(vid2.get(3))
hgt2 = int(vid2.get(4))

bar_wdt = 30

if fps1 == fps2:
    print('matched')
    print(fps1)

wdt = (wdt1 + wdt2)# + bar_wdt

if hgt1 > hgt2:
    hgt = hgt1
else:
    hgt = hgt2

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_vid = cv2.VideoWriter(OUTPUT_VID, fourcc, fps1, (wdt, hgt))
            
try:
#start reading
    while True:
        return_value1, frame1 = vid1.read()
        return_value2, frame2 = vid2.read()
        # frame1 = cv2.resize(wdt1,hgt1)
        # frame2 = cv2.resize(wdt1, hgt1)
        #could cv2.resize
        if return_value1 and return_value2:
            #column = np.zeros(shape = [hgt,bar_wdt,3], dtype = np.uint8)
            #frame1.copyTo(result)
            # frame1 = np.asarray(frame1)
            # frame2 = np.asarray(frame2)
            result = np.concatenate((frame1,frame2), axis = 1)#column,frame2), axis = 1)
            # result.resize(wdt1 *2, hgt1)
            #result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            out_vid.write(result)           
        else:
            cv2.destroyWindow('result')
            print('Video has ended')
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            
        #end video, close viewer, stop writing to file
    vid1.release()
    vid2.release()
    try:
        out_vid.release()
    except:
        pass
    cv2.destroyAllWindows()
    
except:
    print("Unexpected error:", sys.exc_info()[0])
    vid1.release()
    vid2.release()
    try:
        out_vid.release()
    except:
        pass
    cv2.destroyAllWindows()
