# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:42:24 2020

@author: Nikki
"""


import numpy as np
import cv2
import transform as tform
import sys
import math
import scipy.spatial
import markers


###---------------------------------------------------------------------------
#   Given photo points at people's feet, draws '6 foot' ellipse around them.
#   Most useful of these functions for implementing with yolo bounding box points.
#
#   returns - img - input frame with ellipses drawn at specified points
###

def draw_radius(frame, pix_pts, real_pix, pix_real, real_origin):
    bounds, frame = four_pts(pix_pts, pix_real, real_pix, real_origin, frame)
    mytree = load_tree(pix_pts, pix_real)
    img, count = draw_ellipse(frame, bounds, pix_pts, mytree, pix_real)
    # bird_img = overhead(pix_pts, real_origin, pix_real)
    return img, count



###---------------------------------------------------------------------------
#   Given an array of photo pts and conversion matrices, converts to GPS, finds
#   defining points of 6 ft circle at camera angle, and converts back to pixel coords.
#
#   returns - final - array of arrays of 4 pixel coordinates to be used to define each ellipse's axes
###

def four_pts(pix_pts, pix_real, real_pix, real_origin, img):
    #convert to gps coords
    real_pts = tform.transform_pt_array(pix_pts, pix_real)
    final = []
    
    #calculate locations six feet away at given bearings and add to array
    for pt in real_pts:
        degrees = calc_angle(pt, real_origin)
        # print(degrees)
        for angle in degrees:
            location = six_ft(pt, angle)            
            final.append(location)
            
    #convert list of pts to numpy array
    final = np.array([final])
    final = np.squeeze(np.asarray(final))
    #check if final has any elements?
    #convert to pixel coords
    # pt = final[5]
    # pt2 = final[7]

    # dist = math.sqrt((pt2[0] - pt[0])**2 + (pt2[1] - pt[1])**2)
    # print(dist)
    
    
    final = tform.transform_pt_array(final, real_pix)
    
    #show edge circles or not
    # for pt in final:
        
    #     pt2 = (int(pt[0]), int(pt[1]))
    #     cv2.circle(img, pt2, 10, (0,0,255), -1, 8)

    
    
    
    return final, img



###---------------------------------------------------------------------------
#   Given a point, calculates it's bearing in relation to the approximate camera location.
#   This enables GPS circle points to be found such that they define an ellipse within pixel
#   plane that appears properly scaled. Uses haversine formula.
#   Formula from: https://www.movable-type.co.uk/scripts/latlong.html
#   
#   returns - array of 4 bearings in degrees, clockwise from north. First is bearing 
#             between camera and given pt)
###
        
def calc_angle(pt, real_origin):
    #find angle between two points
    # print(pt)
    # print(real_origin)
    opp = (pt[1] - real_origin[1])
    adj = (pt[0] - real_origin[0])
    # print(opp)
    # print(adj)
    a = math.atan(opp/adj)
    # print(a)
    
    a = math.degrees(a)
    # print(a)
    #fill arrray with 90 degree increments
    angle = 4 * [None]
    i = 0
    while i < 4:
        angle[i] = (a + i * 90) % 360
        i = i + 1
    
    return angle



###---------------------------------------------------------------------------
#   Given a GPS point and a bearing, finds point six feet away in that direction,
#   using haversine formula.
#   Formula from: https://www.movable-type.co.uk/scripts/latlong.html
#
#   returns - GPS coord 6 ft away
### 

def six_ft(pt1, a):
    a = math.radians(a)
    #convert to rad
    
    #find x and y component of point 6 ft away
    x = math.cos(a) * 6
    y = math.sin(a) * 6
    
    #add to known point 
    pt2 = pt1 + (x, y)
    
    
    return(pt2)

###---------------------------------------------------------------------------
#   Loads array of pts into a ckd tree for to enable easy finding of nearest pt
#
#   returns - ckd tree
###

def load_tree(pix_pts, pix_real):
    real = tform.transform_pt_array(pix_pts, pix_real)
    mytree = scipy.spatial.cKDTree(real)
      
    return mytree




###---------------------------------------------------------------------------
#   Given array of defining points of several ellipses (endpoints of axes) and 
#   corresponding center points, draws ellipses on given image
#
#   returns - all_img - given image with ellipses drawn onto it
###

def draw_ellipse(frame, pts, centers, mytree, pix_real):
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
    ct = 0
    gps_centers = tform.transform_pt_array(centers, pix_real)
    while i < pts.shape[0]:
        a = pts[i]
        
        b = pts[i + 1]
        c = pts[i + 2]
        d = pts[i + 3]
        # this has performed worse on all samples except maybe aot2
        # possible_minor = int((math.sqrt(math.pow((c[0]-a[0]), 2) + math.pow((c[1]-a[1]), 2)))/2)
        possible_minor = int(abs(c[1]-a[1])/2)
        possible_major = int((math.sqrt(math.pow((d[0]-b[0]), 2) + math.pow((d[1]-b[1]), 2)))/2)
        
        minor = min([possible_major, possible_minor])
        major = max([possible_major, possible_minor])
        
        if centers.size <= 2:
            centers = np.array([centers])
        center = centers[i//4]
        
        x = int(center[0])
        y = int(center[1])
        
        # TODO could probably query all points simultaneously, could be a little more efficient
        if centers.size > 2:
            gps_center = gps_centers[i//4]
            dist, ind = mytree.query(gps_center, k=2)
            closest = mytree.data[ind[1]]
            #dist = GPS_to_ft(gps_center, closest)
            if dist[1] < 6:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 0, 0), thickness, line_type)
                count = count + 1
            elif dist[1] < 8:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 140, 0), thickness, line_type)
            elif dist[1] < 10:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 255, 0), thickness, line_type)            
            else:
                cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (0,255,0), thickness, line_type)
            
            # if ct<4:
            #     cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (0, 0, 255), thickness, line_type)
            # elif ct<8:    
            #     cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (0, 255, 255), thickness, line_type)
            # elif ct<12:
            #     cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (0, 255, 0), thickness, line_type)
            # elif ct<16:
            #     cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (255, 255, 0), thickness, line_type)
            
        else:
            cv2.ellipse(ellipses, (x,y), (major, minor), 0, 0, 360, (0,255,0), thickness, line_type)
        i = i + 4
        ct = ct + 1
        

    #combine original image and ellipse image into one
    all_img = cv2.addWeighted(ellipses, alpha, frame, 1-alpha, 0)
    return all_img, count





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
#   Returns pixel coordinate value of location left-clicked on screen 
#   Based on:
#   https://stackoverflow.com/questions/60066334/get-pixel-coordinates-using-mouse-in-cv2-video-frame-with-python
def get_pixel_coord(video_path):
    try:  
        video_capture = cv2.VideoCapture(video_path)
        
        def mouseHandler(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
        
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("result", mouseHandler)
        
        
        while(True):
        
            # Capture frame-by-frame
            _, frame = video_capture.read()
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            cv2.imshow("result", frame)
        video_capture.release()
        cv2.destroyAllWindows() 
        
    except:
        video_capture.release()
        cv2.destroyAllWindows() 
        
    
    
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
        
def overhead(pts, origin, pix_real):
    a = np.array([2592, 1944])
    b = np.array([2592, 972])
    c = np.array([2592, 0])
    d = np.array([1296, 0])
    e = np.array([0, 0])
    f = np.array([0, 972])     
    g = np.array([0, 1944])
    h = np.array([1296, 1944])
    corners = np.array([a, b, c, d, e, f, g, h])
    real_corners = tform.transform_pt_array(corners, pix_real)
    #find locations of extremes
    real_pts = tform.transform_pt_array(pts, pix_real)
    
    #find most negative values in both directions
    mins = np.ndarray.min(real_corners, axis=0)
    mins[mins > 0] = 0
    mins = np.absolute(mins)
    
    #add to both directions until all values are positive
    shifted = real_corners + mins
    real_pts = real_pts + mins

    #scale frame size
    maxs = np.ndarray.max(shifted, axis=0)
    frame_hgt = int(maxs[0])
    frame_wdt = int(maxs[1])

    #generate blank frame    
    img = np.zeros((frame_hgt, frame_wdt, 3), np.uint8) 

    #draw circles for all included points
    i = 0
    for pt in real_pts:
        x = int(pt[0])
        y = int(pt[1])
        cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
        # if i<4:
        #     cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        # elif i<8:    
        #     cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
        # elif i<12:
        #     cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        # elif i<16:
        #     cv2.circle(img, (x, y), 6, (255, 255, 0), -1)
        # else:
        #     cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
        i = i+1
    origin = origin + mins
    x = int(origin[0]) 
    y = int(origin[1]) 

    
    cv2.circle(img, (x, y), 6, (255, 255, 0), -1)
    return img
        
###---------------------------------------------------------------------------
#   Draws 4 ellipses on video, utilizing most functions in this doc.
###

def test():
    # define where video comes from
    # video_path = './data/AOTsample3.mp4' 
    video_path = 'C:/Users/Nikki/Documents/work/inputs-outputs/video/vid_short.mp4'
    video_path = 'C:/Users/Nikki/Documents/work/inputs-outputs/video/AOTsample1_1.mp4'

    # pt1 = np.array([36.150434, -86.800694])
    # pt2 = np.array([36.150748, -86.800867])
    # pt3 = np.array([36.150617, -86.801132])
    
    # print(GPS_to_ft(pt1, pt2))
    # print(GPS_to_ft(pt2, pt3))
    
    # get transfer function from known GPS and pixel locations

    #AOT 1
    pix_real = [[-1.10159024e-01, -2.79958285e-01,  2.15642254e+02],
                [-1.13855523e-02, -1.39473016e+00,  8.15814671e+02],
                [-2.11104956e-04, -3.64903525e-03,  1.00000000e+00]]
    real_pix = [[ 1.05161691e+01, -3.36930756e+00,  4.81000000e+02],
                [-1.06900637e+00, -4.29603818e-01,  5.81000000e+02],
                [-1.68082662e-03, -2.27891683e-03,  1.00000000e+00]]
    
    pix_real = [[-2.07813620e-01, -5.14012432e-01,  4.01979808e+02],
                [-1.45283091e-16, -3.02228294e+00,  1.72572356e+03],
                [ 4.24715690e-04, -7.70456596e-03,  1.00000000e+00]]
    real_pix = [[ 1.63574796e+01, -4.11269628e+00,  5.22000000e+02],
                [ 1.16697172e+00, -6.02703438e-01,  5.71000000e+02],
                [ 2.04373330e-03, -2.89684039e-03,  1.00000000e+00]]
    
    #vid_short
    # pix_real = [[ 2.91445619e-01,  4.86714183e-01, -2.14894512e+02],
    #             [ 2.36746272e-03,  1.20740599e+00, -4.15252961e+00],
    #             [ 7.42232523e-04,  5.70630790e-03,  1.00000000e+00]]
    # real_pix = [[ 3.51000287e+00, -4.88385701e+00,  7.34000000e+02],
    #             [-1.55374099e-02,  1.28569924e+00,  2.00000000e+00],
    #             [-2.51657708e-03, -3.71163815e-03,  1.00000000e+00]]
    
    # load in sample pts
    # a = np.array([36.148342, -86.799332])   #closest lamp
    # b = np.array([36.148139, -86.799375])   #lamp across street, right
    # c = np.array([36.148349, -86.799135])   #closest left corner of furthest crosswalk dash to right
    # d = np.array([36.147740, -86.799218])   #sixth tree down the street
    
    a = np.array([1296, 1944/6*5])   #far left street pole
    b = np.array([1296, 1944/6*4])   #pole by bike sign
    c = np.array([1296, 1944/6*3])   #corner of sidewalk
    d = np.array([1296, 1944/6*2])   #right of sidewalk stripe closest to camera
    
    e = np.array([1296/2, 1944/6*2]) 
    f = np.array([1296/2, 1944/6*3]) 
    g = np.array([1296/2, 1944/6*4]) 
    h = np.array([1296/2, 1944/6*5]) 
    
    i = np.array([1296/2*3, 1944/6*2]) 
    j = np.array([1296/2*3, 1944/6*3]) 
    k = np.array([1296/2*3, 1944/6*4]) 
    l = np.array([1296/2*3, 1944/6*5]) 
    
    m = np.array([1296/4*3, 1944/6*2]) 
    n = np.array([1296/4*3, 1944/6*3]) 
    o = np.array([1296/4*3, 1944/6*4]) 
    p = np.array([1296/4*3, 1944/6*5])
    orig = np.array([1296, 1944])
    #orig = np.array([1296*2, 0])
    
    
    
    # a = np.array([1280/2, 720/6*5])   #far left street pole
    # b = np.array([1280/2, 720/6*4])   #pole by bike sign
    # c = np.array([1280/2, 720/6*3])   #corner of sidewalk
    # d = np.array([1280/2, 720/6*2])   #right of sidewalk stripe closest to camera
    
    # e = np.array([1280/2/2, 720/6*2]) 
    # f = np.array([1280/2/2, 720/6*3]) 
    # g = np.array([1280/2/2, 720/6*4]) 
    # h = np.array([1280/2/2, 720/6*5]) 
    
    # i = np.array([1280/2/2*3, 720/6*2]) 
    # j = np.array([1280/2/2*3, 720/6*3]) 
    # k = np.array([1280/2/2*3, 720/6*4]) 
    # l = np.array([1280/2/2*3, 720/6*5]) 
    
    # m = np.array([1280/2/4*3, 720/6*2]) 
    # n = np.array([1280/2/4*3, 720/6*3]) 
    # o = np.array([1280/2/4*3, 720/6*4]) 
    # p = np.array([1280/2/4*3, 720/6*5])
    # orig = np.array([1280/2, 720])
    #d = np.array([1296, 1944])
    x = np.array([a,b,c,d, e, f, g, h, i, j, k, l, m, n, o, p, orig])
    
    pix_pts = x
    
    # start video
    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)
    wdt = int(vid.get(3))
    hgt = int(vid.get(4))
    print (wdt)
    print(hgt)
    
    origin = orig
    print(origin)
    real_origin = tform.transform_pt_array(origin, pix_real)
    real_origin = np.squeeze(np.asarray(origin))
    print(real_origin)
    
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
            img, count = draw_radius(frame, pix_pts, real_pix, pix_real, real_origin)
            pt = (int(orig[0]), int(orig[1]))
            cv2.circle(img, pt, 20, (0,0,255), -1, 8)
            
            img2 = overhead(pix_pts, origin, pix_real)

            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", img)
            cv2.namedWindow("Overhead", cv2.WINDOW_NORMAL)
            cv2.imshow("Overhead", img2)
            if cv2.waitKey(1000) & 0xFF == ord('q'): break
    
        # end video, close viewer, stop writing to file
        vid.release()
        cv2.destroyAllWindows()
    
    # if interrupted, end video, close viewer, stop writing to file
    except:
        print("Unexpected error:", sys.exc_info())
        vid.release()
        cv2.destroyAllWindows()
     
#test()

# pt1 = np.array([36.150319, -86.801259])
# pt2 = np.array([36.150258, -86.801385])
# pt3 = np.array([36.150490, -86.801388])
# pt4 = np.array([36.150427, -86.801514])


# print('width ' + str(GPS_to_ft(pt1, pt2)))
# print('width ' + str(GPS_to_ft(pt3, pt4)))
# print('height ' + str(GPS_to_ft(pt1, pt3)))
# print('height ' + str(GPS_to_ft(pt2, pt4)))
