# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:14:45 2020

@author: Nikki
"""

import numpy as np
import cv2
import transform as tform
import sys
import math
import scipy.spatial

###---------------------------------------------------------------------------
#   Allows video to be initialized using a string
#
#   returns - video_path - path to video to be used
#   returns - GPS_pix - matrix to convert from GPS to pixel
#           - pix_GPS - matrix to convert from pixel to GPS
#           - origin - approximate camera location in GPS 
###

def sample_select(name):        
    if name == 'aot3':
        video_path = './data/video/AOTsample3.mp4'
    elif name == 'mrb3':
        video_path = './data/video/20190422_153844_DA4A.mkv'
    elif name == 'aot1':
        video_path = './data/video/AOTsample1_1.mp4'
    elif name == 'aot2':
        video_path = './data/video/AOTsample2_1.mp4'
        
    GPS_pix, pix_GPS, origin = get_transform(name)

    return video_path, GPS_pix, pix_GPS, origin
        



###---------------------------------------------------------------------------
#   Used to find transformation matrices between GPS and pixel space and vice versa.
#
#   returns - GPS_pix - matrix to convert from GPS to pixel
#           - pix_GPS - matrix to convert from pixel to GPS
###

def get_transform(name):
    if name == 'aot3':
        x, y, origin = aot_3_markers()
        
    elif name == 'mrb3':
        x, y, origin = mrb3_markers()
        
    elif name == 'aot1':
        x, y, origin = aot_1_markers()
    
    elif name == 'aot2':
        x, y, origin = aot_2_markers()
        
    GPS_pix = tform.get_best_transform(x, y)
    pix_GPS = tform.get_best_transform(y, x)
    
    return(GPS_pix, pix_GPS, origin)



###---------------------------------------------------------------------------
#   Contains GPS and pixel information for AOT on 21 Ave
#
#   returns - x - GPS location of 4 pts
#           - y - pixel locations of corresponding 4 pts
#           - origin - approximate camera location in GPS

def aot_3_markers():
    #get transfer function from known GPS and pixel locations
    a = np.array([36.148342, -86.799332])   #closest lamp
    b = np.array([36.148139, -86.799375])   #lamp across street, right
    c = np.array([36.148349, -86.799135])   #closest left corner of furthest crosswalk dash to right
    d = np.array([36.147740, -86.799218])   #sixth tree down the street
    #d = np.array([36.148248, -86.799228])   #fifth turning dash
    #a1 = np.array([36.148375, -86.799294])   #close front edge of stopping traffic line on left
    #b1 = np.array([36.148369, -86.799229])   #far front edge of stopping traffic line on left
    
    e = np.array([1658, 1406])
    f = np.array([2493, 1190])
    g = np.array([492, 990])
    h = np.array([2290, 970])
    #h = np.array([1481, 1090])
    #e1 = np.array([992, 1386])
    #f1 = np.array([667, 1166])
    
    
    x = np.array([a,b,c,d])
    y = np.array([e,f,g,h])
    
    #approx camera location aot_3
    origin = np.array([36.148432, -86.799378])
    
    return (x, y, origin)



###---------------------------------------------------------------------------
#   Contains GPS and pixel information for AOT on 21 Ave

def aot_2_markers():
    #get transfer function from known GPS and pixel locations
    a = np.array([36.150310, -86.801245])   #far left corner of yellow strip before crosswalk
    b = np.array([36.150623, -86.801172])   #front right corner of DGX
    c = np.array([36.150479, -86.801590])   #Ruth's Chris leftmost door under tent
    d = np.array([36.150239, -86.801372])   #front of line in sidewalk
    
    e = np.array([2227, 1416])
    f = np.array([2234, 789])
    g = np.array([362, 758])
    h = np.array([220, 1679])

    
    
    x = np.array([a,b,c,d])
    y = np.array([e,f,g,h])
    
    #approx camera location aot_3
    origin = np.array([36.150190, -86.801302])
    
    return (x, y, origin)

###---------------------------------------------------------------------------
#   Contains GPS and pixel information for AOT on 21 Ave

def aot_1_markers():
    #get transfer function from known GPS and pixel locations
    a = np.array([36.150191, -86.801195])   #edge of brick curve closest to center
    b = np.array([36.150494, -86.800601])   #base of sign in between cars closer to qdoba building
    c = np.array([36.150951, -86.800553])   #corner of t-mobile store
    d = np.array([36.150310, -86.801245])   #far left corner of yellow strip before crosswalk

    e = np.array([2227, 1547])
    f = np.array([1640, 544])
    g = np.array([752, 524])
    h = np.array([137, 1370])

    
    x = np.array([a,b,c,d])
    y = np.array([e,f,g,h])
    
    #approx camera location aot_1
    origin = np.array([36.150190, -86.801302])
    
    return (x, y, origin)



###---------------------------------------------------------------------------
#   Contains GPS and pixel information for mrb3 camera

def mrb3_markers():
    #get transfer function from known GPS and pixel locations
    a = np.array([36.144187, -86.799707])   #far left street pole
    b = np.array([36.143990, -86.799594])   #pole by bike sign
    c = np.array([36.143997, -86.800180])   #corner of sidewalk
    d = np.array([36.144203, -86.800149])   #right of sidewalk stripe closest to camera
      
    e = np.array([18, 1151])
    f = np.array([462, 210])
    g = np.array([3286, 749])
    h = np.array([2940, 2150])
    
    x = np.array([a,b,c,d])
    y = np.array([e,f,g,h])
    
    #approx cam location mrb 3
    origin = np.array([36.144322, -86.800059])
    return (x, y, origin)



###---------------------------------------------------------------------------
#   Given photo points at people's feet, draws '6 foot' ellipse around them.
#   Most useful of these functions for implementing with yolo bounding box points.
#
#   returns - img - input frame with ellipses drawn at specified points
###

def draw_radius(frame, pts, GPS_pix, pix_GPS, origin):

    bounds = four_pts(pts, pix_GPS, GPS_pix, origin)
    mytree = load_tree(pts, pix_GPS)
    img, count = draw_ellipse(frame, bounds, pts, mytree, pix_GPS)
    return img, count



###---------------------------------------------------------------------------
#   Given an array of photo pts and conversion matrices, converts to GPS, finds
#   defining points of 6 ft circle at camera angle, and converts back to pixel coords.
#
#   returns - final - array of arrays of 4 pixel coordinates to be used to define each ellipse's axes
###

def four_pts(pts, pix_GPS, GPS_pix, origin):
    
    #convert to gps coords
    gps = tform.transform_pt_array(pts, pix_GPS)
    final = []
    
    #calculate locations six feet away at given bearings and add to array
    for pt in gps:
        degrees = calc_bearing(pt, origin)
        for angle in degrees:
            a = six_ft(pt, angle)
            final.append(a)
    #convert list of pts to numpy array
    final = np.array([final])
    final = np.squeeze(np.asarray(final))
    
    #check if final has any elements?
    #convert to pixel coords
    final = tform.transform_pt_array(final, GPS_pix)
    return final



###---------------------------------------------------------------------------
#   Given a point, calculates it's bearing in relation to the approximate camera location.
#   This enables GPS circle points to be found such that they define an ellipse within pixel
#   plane that appears properly scaled. Uses haversine formula.
#   Formula from: https://www.movable-type.co.uk/scripts/latlong.html
#   
#   returns - array of 4 bearings in degrees, clockwise from north. First is bearing 
#             between camera and given pt)
###
        
def calc_bearing(pt, origin):

    
    #convert GPS coords to radians
    la1 = math.radians(origin[0])
    la2 = math.radians(pt[0])
    lo1 = math.radians(origin[1])
    lo2 = math.radians(pt[1])
    
    #perform calculation
    y = math.sin(lo2-lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2-lo1)
    b = math.atan2(y,x)
    
    #convert to degrees
    b = math.degrees(b)
    
    #fill arrray with 90 degree increments
    bearing = 4 * [None]
    i = 0
    while i < 4:
        bearing[i] = (b + i * 90) % 360
        i = i + 1
    
    return bearing

###---------------------------------------------------------------------------
#   Loads array of pts into a ckd tree for to enable easy finding of nearest pt
#
#   returns - ckd tree
###

def load_tree(pts, pix_GPS):
    gps = tform.transform_pt_array(pts, pix_GPS)
    mytree = scipy.spatial.cKDTree(gps)
      
    return mytree



###---------------------------------------------------------------------------
#   Given array of defining points of several ellipses (endpoints of axes) and 
#   corresponding center points, draws ellipses on given image
#
#   returns - all_img - given image with ellipses drawn onto it
###

def draw_ellipse(frame, pts, centers, mytree, pix_GPS):
    #define qualities of the ellipse
    thickness = -1
    line_type = 8
    
    #set transparency
    alpha = 0.25
    
    #create separate image for ellipses to be drawn into
    ellipses = frame.copy()
    
    #iterate through list of ellipse points and centers, drawing each into ellipse image    
    i = 0
    count = 0
    gps_centers = tform.transform_pt_array(centers, pix_GPS)
    while i < pts.shape[0]:
        a = pts[i]
        b = pts[i + 1]
        c = pts[i + 2]
        d = pts[i + 3]
        minor = int((math.sqrt(math.pow((c[0]-a[0]), 2) + math.pow((c[1]-a[1]), 2)))/2)
        major = int((math.sqrt(math.pow((d[0]-b[0]), 2) + math.pow((d[1]-b[1]), 2)))/2)
        
        if centers.size <= 2:
            centers = np.array([centers])
        center = centers[i//4]
        
        x = int(center[0])
        y = int(center[1])
        
        if centers.size > 2:
            gps_center = gps_centers[i//4]
            dist, ind = mytree.query(gps_center, k=2)
            closest = mytree.data[ind[1]]
            dist = GPS_to_ft(gps_center, closest)
            if dist < 6:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 0, 0), thickness, line_type)
                count = count + 1
            elif dist < 8:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 140, 0), thickness, line_type)
            elif dist < 10:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 255, 0), thickness, line_type)            
            else:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (0,255,0), thickness, line_type)
        else:
            cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (0,255,0), thickness, line_type)
        i = i + 4
    #combine original image and ellipse image into one
    all_img = cv2.addWeighted(ellipses, alpha, frame, 1-alpha, 0)
    return all_img, count



###---------------------------------------------------------------------------
#   Given a GPS point and a bearing, finds point six feet away in that direction,
#   using haversine formula.
#   Formula from: https://www.movable-type.co.uk/scripts/latlong.html
#
#   returns - GPS coord 6 ft away
### 

def six_ft(pt1, b):
    
    #convert to rad
    la1 = math.radians(pt1[0])
    lo1 = math.radians(pt1[1])
    b = math.radians(b)
    
    #calc latitude and longitude
    radius = 20902231
    d =(6.0/radius)
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2((math.sin(b) * math.sin(d) * math.cos(la1)), (math.cos(d) - math.sin(la1) * math.sin(la2)))
    
    #reconvert to GPS standard, degrees
    pt2 = (math.degrees(la2), math.degrees(lo2))
    
    return(pt2) 

###---------------------------------------------------------------------------
#   Given two GPS points, finds distance in ft between them, calulated using 
#   haversine formula. 
#
#   returns - distance in ft between given points
###

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





###---------------------------------------------------------------------------
#   Following functions are not utilized in video processing code, but were helpful
#   during development
###---------------------------------------------------------------------------


###---------------------------------------------------------------------------
#   Given points, draws circles around them 
###
      
def make_circles(frame, centers, size):
    size = size[0] // 128
    thickness = -1
    line_type = 8
    for center in centers:
        pt = (int(center[0]), int(center[1]))
        cv2.circle(frame, pt, size, (0,0,255), thickness, line_type)
        
        
        
###---------------------------------------------------------------------------
#   Draws 4 ellipses on video, utilizing most functions in this doc.
###

def test():
    # define where video comes from
    # video_path = './data/AOTsample3.mp4' 
    video_path = './data/20190422_153844_DA4A.mkv'
    
    # get transfer function from known GPS and pixel locations
    GPS_pix, pix_GPS = get_transform()
    
    
    # load in sample pts
    # a = np.array([36.148342, -86.799332])   #closest lamp
    # b = np.array([36.148139, -86.799375])   #lamp across street, right
    # c = np.array([36.148349, -86.799135])   #closest left corner of furthest crosswalk dash to right
    # d = np.array([36.147740, -86.799218])   #sixth tree down the street
    
    a = np.array([36.144187, -86.799707])   #far left street pole
    b = np.array([36.143990, -86.799594])   #pole by bike sign
    c = np.array([36.143997, -86.800180])   #corner of sidewalk
    d = np.array([36.144203, -86.800149])   #right of sidewalk stripe closest to camera
    x = np.array([a,b,c,d])
    
    pts = tform.transform_pt_array(x, GPS_pix)
    print(pts)
    
    # start video
    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)
    try:
        while True:
            # skip desired number of frames to speed up processing
            for i in range (10):
                vid.grab()
            
            # read frame
            return_value, frame = vid.read()
            # if frame doesn't exist, exit
            if not return_value:
                cv2.destroyWindow('result')
                print('Video has ended')
                break
            
            # draw ellipse
            img, count = draw_radius(frame, pts, GPS_pix, pix_GPS)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    
        # end video, close viewer, stop writing to file
        vid.release()
        cv2.destroyAllWindows()
    
    # if interrupted, end video, close viewer, stop writing to file
    except:
        print("Unexpected error:", sys.exc_info()[0])
        vid.release()
        cv2.destroyAllWindows()
 
#test()