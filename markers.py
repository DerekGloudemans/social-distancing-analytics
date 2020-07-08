import numpy as np


###---------------------------------------------------------------------------
#   Contains GPS and pixel information for AOT on 21 Ave
#
#   returns - x - GPS location of 4 pts
#           - y - pixel locations of corresponding 4 pts
#           - origin - approximate camera location in GPS

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
    
    #approx camera location aot_2
    origin = np.array([36.150190, -86.801302])
    
    return (x, y, origin)


###---------------------------------------------------------------------------
#   Contains GPS and pixel information for AOT on 21 Ave
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
