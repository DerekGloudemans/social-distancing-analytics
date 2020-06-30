# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:14:45 2020

@author: Nikki
"""

import numpy as np
import cv2
import transform as tran
from PIL import Image
import sys
import math
# def main():
#     #define where video comes from
#     video_path = './data/AOTsample3.mp4' 
    
#     #get transfer function from known GPS and pixel locations
#     GPS_pix, pix_GPS = get_transform()
    
#     #GPS_to_ft((36.148342, -86.799332), (36.148139, -86.799375))
    
#     #known pixel locations
#     pts = tran.transform_pt_array(x, GPS_pix)
#     # pts2 = tran.transform_pt_array(y, pix_GPS)
#     # pts3 = tran.transform_pt_array(pts2, GPS_pix)
#     # print(pts)
#     # print(pts2)
#     # print(pts3)
    
#     six_ft(a, 90)
            
#     #start video
#     print("Video from: ", video_path )
#     vid = cv2.VideoCapture(video_path)
    
#     try:
                
#         while True:
#             #skip desired number of frames to speed up processing
#             for i in range (10):
#                 vid.grab()
            
#             #read frame
#             return_value, frame = vid.read()
#             frame_size = frame.shape[:2]
            
#             # if frame doesn't exist, exit
#             if not return_value:
#                 cv2.destroyWindow('result')
#                 print('Video has ended')
#                 break
            
#             #draw circles at pixel locations
#             make_circles(frame, pts, frame_size)
#             # make_circles(frame, pts3, frame_size)
#             #draw extra circles
#             draw_ellipse(frame, four_pts(pts, pix_GPS, GPS_pix))
#             #make_circles(frame, four_pts(pts, pix_GPS, GPS_pix), frame_size)
                
#             cv2.namedWindow("result", cv2.WINDOW_NORMAL)
#             cv2.imshow("result", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'): break
    
#         #end video, close viewer, stop writing to file
#         vid.release()
#         cv2.destroyAllWindows()
#         f.close()
    
    
#     #if interrupted, end video, close viewer, stop writing to file
#     except:
#         print("Unexpected error:", sys.exc_info()[0])
#         vid.release()
#         cv2.destroyAllWindows()
        
##given points, draws a cirle where they should be        
def make_circles(frame, centers, size):
    size = size[0] // 128
    thickness = -1
    line_type = 8
    for center in centers:
        pt = (int(center[0]), int(center[1]))
        cv2.circle(frame, pt, size, (0,0,255), thickness, line_type)
        
        

#using haversine formula    
def GPS_to_ft(pt1, pt2):
    #earths rad in ft
    radius = 20902231
    la1 = math.radians(pt1[0])
    la2 = math.radians(pt2[0])
    lo1 = math.radians(pt1[1])
    lo2 = math.radians(pt2[1])
    
    #la2, lo2 = six_ft(pt1, 90)
    a = math.pow(((la2 - la1) / 2), 2)
    b = math.cos(la1) * math.cos(la2)
    c = math.pow(((lo2 - lo1) / 2), 2)
    d = math.sin(a) + b * math.sin(c)
    
    dist = 2 * radius * math.asin(math.sqrt(d))
    #print(dist)
    return dist


#returns GPS coord that is 6 ft away    
def six_ft(pt1, b):
    #print("six")
    la1 = math.radians(pt1[0])
    #la2 = math.radians(pt2[0])
    lo1 = math.radians(pt1[1])
    #lo2 = math.radians(pt2[1])
    b = math.radians(b)
    radius = 20902231
    d =(6.0/radius)
    #b = math.radians(90)
    
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2((math.sin(b) * math.sin(d) * math.cos(la1)), (math.cos(d) - math.sin(la1) * math.sin(la2)))
    
    #reconvert to GPS standard, degrees
    pt2 = (math.degrees(la2), math.degrees(lo2))
    
    # double check that coords are 6 ft apart
    # GPS_to_ft(pt1, pt2)
    return(pt2)    

#given an array of photo pts, returns list of four pts in circles around them
def four_pts(pts, pix_GPS, GPS_pix):
    #print("four")
    
    #convert to gps coords
    gps = tran.transform_pt_array(pts, pix_GPS)
    
    #bearing measurements--degrees clockwise from north (for aot3, would guess ~145)
    
    degrees = [55, 145, 235, 325]
    
    final = []
    
    #calculate locations six feet away and add to array
    for pt in gps:
        degrees = calc_bearing(pt)
        for angle in degrees:
            a = six_ft(pt, angle)
            final.append(a)
   
    #convert list of pts to numpy array
    final = np.array([final])
    final = np.squeeze(np.asarray(final))

    #convert to pixel coords
    final = tran.transform_pt_array(final, GPS_pix)
    #print(final)
    return final

def draw_ellipse(frame, pts, centers):
    #print("ell")
    i = 0
    thickness = -1
    line_type = 8
    
    ellipses = frame.copy()
    alpha = 0.5
    while i < pts.shape[0]:
        a = pts[i]
        b = pts[i + 1]
        c = pts[i + 2]
        d = pts[i + 3]
        minor = int((math.sqrt(math.pow((c[0]-a[0]), 2) + math.pow((c[1]-a[1]), 2)))/2)
        major = int((math.sqrt(math.pow((d[0]-b[0]), 2) + math.pow((d[1]-b[1]), 2)))/2)
        index = i//4
        center = centers[index]#a + (c-a)/2
        x = int(center[0])
        y = int(center[1])
        
        cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 0,255), thickness, line_type)
        
        i = i + 4
        
    all_img = cv2.addWeighted(ellipses, alpha, frame, 1-alpha, 0)
    return all_img
        
def calc_bearing(pt):
    #approx camera location
    origin = np.array([36.148432, -86.799378])
    la1 = math.radians(origin[0])
    la2 = math.radians(pt[0])
    lo1 = math.radians(origin[1])
    lo2 = math.radians(pt[1])
    y = math.sin(lo2-lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2-lo1)
    b = math.atan2(y,x)
    b = math.degrees(b)
    #print("bear:", b)
    bearing = 4 * [None]
    i = 0
    while i < 4:
        bearing[i] = (b + i * 90) % 360
        i = i + 1
    
    #print(bearing)
    return bearing

def get_transform():
        
    #get transfer function from known GPS and pixel locations
    a = np.array([36.148342, -86.799332])   #closest lamp
    b = np.array([36.148139, -86.799375])   #lamp across street, right
    c = np.array([36.148349, -86.799135])   #closest left corner of furthest crosswalk dash to right
    #d = np.array([36.148248, -86.799228])   #fifth turning dash
    #a1 = np.array([36.148375, -86.799294])   #close front edge of stopping traffic line on left
    #b1 = np.array([36.148369, -86.799229])   #far front edge of stopping traffic line on left
    c1 = np.array([36.147740, -86.799218])   #sixth tree down the street
    # d1 = np.array([36.148248, -86.799228])   #fifth turning dash
    
    e = np.array([1658, 1406])
    f = np.array([2493, 1190])
    g = np.array([492, 990])
    #h = np.array([1481, 1090])
    #e1 = np.array([992, 1386])
    #f1 = np.array([667, 1166])
    g1 = np.array([2290, 970])
    
    x = np.array([a,b,c,c1])
    y = np.array([e,f,g,g1])
    
    GPS_pix = tran.get_best_transform(x, y)
    pix_GPS = tran.get_best_transform(y, x)
    
    return(GPS_pix, pix_GPS)

# given photo points at people's feet, draws '6 foot' ellipse around them
def draw_radius(frame, pts, GPS_pix, pix_GPS):
    bounds = four_pts(pts, pix_GPS, GPS_pix)
    #print(bounds)
    #print(pts)
    img = draw_ellipse(frame, bounds, pts)
    return img
    
# if __name__ == "__main__":
#     main()    
