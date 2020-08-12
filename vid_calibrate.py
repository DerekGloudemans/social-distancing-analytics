# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:43:34 2020

@author: Nikki
"""

#referencing https://github.com/aqeelanwar/SocialDistancingAI

import cv2
import sys
import csv
import transform as tform
import numpy as np
import os.path


#   Goal Input: csv file with ip address, length, width

#   Output: csv file with ip + transform
def main():

    video_path = 'C:/Users/Nikki/Documents/work/inputs-outputs/video/AOTsample1_1.mp4'
    video_path = 'C:/Users/Nikki/Documents/work/inputs-outputs/video/AOTsample2_1.mp4'
    # video_path = 'rtsp://root:worklab@192.168.86.246/axis-media/media.amp?framerate=30.0?streamprofile=vlc' #hall
    # video_path = 'rtsp://root:worklab@192.168.86.247/axis-media/media.amp?framerate=30.0?streamprofile=vlc' #living room

    # video_path = 'C:/Users/Nikki/Documents/work/inputs-outputs/video/vid_short.mp4'

    output_file = 'C:/Users/Nikki/Documents/work/inputs-outputs/transforms.csv'
    
    #whether to include surroundings, or just the space between the four selected points
    roi_only = True 
    
    # #width and length of measured area, in feet aot 1
    # length = 125.37564416986191
    # wth = 91.52952303334027
  
    # #width and length of measured area, in feet aot 2
    # length = 72.6
    # wth = 43.4

    # living room
    length = 11.18
    wth = 15.64
    
    # #hall
    # length = 19.71
    # wth = 3.375
    
    #enter points top left, top right, bottom left bottom right, 2 * 6 ft apart - should be rectangle
    #furthest locations away and closest locations to camera that someone could be standing
    parallel, corners = find_all_pts(video_path, roi_only)     
    
    #find transfromation matrices
    pix_real, real_pix = find_transform(parallel, corners, wth, length, roi_only)
    
    # pix_real = np.array2string(pix_real, separator = ',')
    # real_pix = np.array2string(real_pix, separator = ',')
    pix_real = pix_real.tolist()
    real_pix = real_pix.tolist()
   
    #if file doesn't exist, create it. Otherwise, append to it
    try:
        if not os.path.isfile(output_file):
            csvfile = open(output_file, 'w+')
        
        else:
            csvfile = open(output_file, 'a+', newline = '')
        writer = csv.writer(csvfile)
        writer.writerow([video_path, pix_real, real_pix])
        csvfile.close()
    except:
        print("Unexpected error:", sys.exc_info())
        csvfile.close()

        
#   --------------------------------------------------------------------------
#   Given four points, and real world distance between them (wth and length), 
#   finds transformation matrix

def find_transform(parallel, corners, wth, length, roi_only = True, display = False):
    
    #convert to numpy arrays, define overhead plane using measurements taken of the area
    parallel = np.array(parallel)
    corners = np.array(corners)
    overhead_pts = np.array([[0,0], [wth, 0], [wth,length], [0, length]])
    
    #get transformation arrays
    pix_real = tform.get_best_transform(parallel, overhead_pts)
    real_pix = tform.get_best_transform(overhead_pts, parallel)
    
    
    if display:
        show_overhead(parallel, corners, pix_real, length, wth, roi_only)
        
    return pix_real, real_pix
            

#   --------------------------------------------------------------------------
#   Displays first frame of video and allows user to click on 4 or 8 pts, which
#   are appended to a list
    
def find_all_pts(video_path, roi_only = True):
    video_capture = cv2.VideoCapture(video_path)
    try:
        
        global allpts
        allpts = []
        _, frame = video_capture.read()
        # use = False
        
        # FIXME cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

        # # Allow user to regrab frame if previous frame was not acceptable
        # while not use:
            
        #     try:
                
        #         key = input('Get new frame (y/n)?')
        #         if key == 'y':
        #             video_capture.grab()
        #             _, frame = video_capture.retrieve()
                    
                    
        #         elif key == 'n':
        #             use = True
        #         else:
        #             print('Invalid option')
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        #         cv2.imshow("Calibration", frame)
        #         if cv2.waitKey(1) & 0xFF == ord('q'): break
        #     except:
        #         break
        
        frame_size = frame.shape[:2]
        corners = [[0,0], [frame_size[1], 0], [frame_size[1],frame_size[0]], [0, frame_size[0]]]
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", get_pix_coord, frame)
        cv2.imshow("Calibration", frame)
       
        if roi_only == True:
            cutoff = 4
        else:
            cutoff = 8
       
        #enter points top left, top right, bottom left bottom right, 2 * 6 ft apar
        while len(allpts) < cutoff:
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            cv2.imshow("Calibration", frame)
        
        
        parallel = allpts[:4]
        corners = allpts[4:]


        video_capture.release()
        cv2.destroyAllWindows()
        
        return parallel, corners
    except:
        print("Unexpected error:", sys.exc_info())
        video_capture.release()
        cv2.destroyAllWindows() 
        
        
#   --------------------------------------------------------------------------
#   Handler to append left-click location to list of points

def get_pix_coord(event, x, y, flags, frame):
    global allpts
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
        if "allpts" not in globals():
            allpts = []
        allpts.append([x,y])
        print("Point selected: " + str(x) + ", " + str(y))


#   --------------------------------------------------------------------------
#   Given points, frame size, and transformation, diplays overhead view of points

def show_overhead(parallel, corners, pix_real, length, wth, roi_only):
    pts = tform.transform_pt_array(parallel, pix_real)
        
    #scale frame size
    if roi_only:
        frame_hgt = length
        frame_wdt = wth
        
        
    #calculate points at extremeties
    else:
        #find locations of extremes
        extremes = tform.transform_pt_array(corners, pix_real)
        
        #find most negative values in both directions
        mins = np.ndarray.min(extremes, axis=0)
        mins[mins > 0] = 0
        mins = np.absolute(mins)
        
        #add to both directions until all values are positive
        shifted = extremes + mins
        
        #scale frame size
        maxs = np.ndarray.max(shifted, axis=0)
        pts = pts + mins
        frame_hgt = int(maxs[1])
        frame_wdt = int(maxs[0])
    
    #generate blank frame    
    img = np.zeros((frame_hgt, frame_wdt, 3), np.uint8) 
    
    #draw circles for all included points
    if not roi_only: 
        for pt in shifted:
             x = int(pt[0])
             y = int(pt[1])
             cv2.circle(img, (x, y), 5, (0, 0, 255), -1)   
            
    for pt in pts:
        x = int(pt[0])
        y = int(pt[1])
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
    
    #display image    
    try:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
    
        cv2.destroyAllWindows()
    except:
        print("Unexpected error:", sys.exc_info())
        cv2.destroyAllWindows()   


        
if __name__ == "__main__":
    main()