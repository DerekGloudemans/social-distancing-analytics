# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:35:56 2020

@author: Nikki
"""
# 1 website only displays live videos and csvs of 1 cam
# 2 website displays live videos + csvs of all
# 3 website displays live individual camera and live aggregate data
# 4 website shows most recent frames
# 5 website shows moving averages + such for hr, day, week
# 6 website shows graphs and such
import sys
import math
import numpy as np

#TODO I think averages are correct, could be good to double check
def main(out_q, buf_num ,num_cams, avgs, avg_lock, errs, ocpts, dists):
    # all_out = [[[None]*3]* buf_num] * num_cams
    # errs = [[None]* buf_num for _ in range(num_cams)]
    # # ocpts = [[None]* buf_num for _ in range(num_cams)]
    # dists = [[None]* buf_num for _ in range(num_cams)]
    # for i in range(num_cams):
    #     errs.append(manager.list([]))
    #     ocpts.append(manager.list([]))
    #     dists.append(manager.list([]))
    index = 0
    rollover = 0
    last = buf_num * -1
    #not sure that a is cycling through the buffers in the best way
    try:
        while(True):
            #location of oldest entry in mega list
            index = index % buf_num
            if not out_q.empty():
                # total = total+1
                rollover = rollover + 1
                data = out_q.get()
                #camera number
                i = data[0]
               
                #updates camera buffer at oldest index
                errs[i].append(data[1])
                ocpts[i].append(data[2])
                dists[i].append(data[3])

                
                while len(errs[i]) > buf_num:
                    errs[i].pop(0)
                
                while len(ocpts[i]) > buf_num:
                    ocpts[i].pop(0)
                    
                while len(dists[i]) > buf_num:
                    dists[i].pop(0)
                
                #this section just for debugging
                # avg_lock.acquire()
                # avgs[0] = str(dists[i])
                # avgs[1] = str(ocpts[i])
            
                # avgs[2] = get_o_avg(ocpts, i)
                # avgs[3] = get_e_avg(errs, i)
                # avgs[4] = get_dist_avg(dists, i)
              
                # avg_lock.release()


                if rollover == (num_cams):
                    rollover = 0
                    index = index + 1
                
    except:
        avgs[0] = 'Error'
        avgs[4] =  str(sys.exc_info())

    return

def get_e_avg(errs, i):
    error_avg = math.ceil(calc_avg(errs[i]))
    return error_avg
    
def get_o_avg(ocpts, i):
    occupant_avg = math.ceil(calc_avg(ocpts[i]))
    return occupant_avg

def get_dist_avg(dists, i):
    dist_avg = round(calc_avg(dists[i]), 2)
    return dist_avg

def total_e_avg(errs):
    error_avg = math.ceil(total_avg(errs))
    return error_avg
    
def total_o_avg(ocpts):
    occupant_avg = math.ceil(total_avg(ocpts))
    return occupant_avg

# def total_dist_avg(dists):
#     dist_avg = round(total_avg(dists), 2)
#     return dist_avg



def calc_avg(num_list):
    total = 0
    count = 0
    for num in num_list:
        if num is not None:
            total = total + num
            count = count + 1
    if count != 0:
        avg = total/count
    else:
        avg = 0
        
    return avg    

def total_avg(stat_list):
    total = 0
    for num_list in stat_list:
        #find approx # of people/errors at each location, then add together for total approximation
        total = total + calc_avg(num_list)
        
    return total 

#want the average distance between people, isn't summed at all
def total_dist_avg(stat_list):
    total = 0
    count = 0
    for num_list in stat_list:
        for num in num_list:
            if num is not None:
                total = total + num
                count = count + 1
    if count != 0:
        avg = total/count
    else:
        avg = 0
    dist_avg = round(avg, 2)    
    return dist_avg  
#buffer of past three frame data stuff?
# realpts, str(time), errors, occupants, avg_dist
#stats = errors, occupants, avg_dist

    

# #### for each camera location + for all combined

# # save to a different csv for later use
# total # people that day
# people/day  
# % complinace/day
#avg dist apart/day

# occupants vs dist apart


#people/hr
#compliance/hr
#avg dist apart/hr


#only generate graphs when clicked on, current data when website is opened
#%compliance/ past day
# people/ past day - > useful for live dashboard
#avg dist apart/ past day

#%compliance/ past hr
# people/ past hr - > useful for live dashboard
#avg dist apart/ past hr



#heatmap generated from people's positions in each location
#hr, day, forever