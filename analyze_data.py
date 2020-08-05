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
import time
import sys
import math
# 6 website shows graphs and such


####FIXME averages are incorrect, WHYYYYY?????
def main(out_q, buf_num ,num_cams, avgs, avg_lock):
    # all_out = [[[None]*3]* buf_num] * num_cams
    errors = [[None]* buf_num for _ in range(num_cams)]
    occupants = [[None]* buf_num for _ in range(num_cams)]
    avg_dists = [[None]* buf_num for _ in range(num_cams)]
    index = 0
    rollover = 0
    total = 0
    #not sure that a is cycling through the buffers in the best way
    # time.sleep(10)
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
                errors[i][index] = data[1]
                occupants[i][index] = data[2]
                avg_dists[i][index] = data[3]
                
                
                # temp = filter(None, errors)
                # num = len(temp)
                # error_avg = sum(temp)/num
                error_avg = math.ceil(calc_avg(errors[i]))
                occupant_avg = math.ceil(calc_avg(occupants[i]))
                dist_avg = round(calc_avg(avg_dists[i]), 2)
                
                #this section just for debugging
                avg_lock.acquire()
                avgs[0] = i
                avgs[1] = index
            
                avgs[2] = occupant_avg
                avgs[3] = error_avg
                avgs[4] = dist_avg
                
                
                
                
                avg_lock.release()
                # avgs[2] = avg_dists[i]
                # avgs[3] = data
                # avgs[4] = [index, a]
                # err_avg = 0
                # occupant_avg = 0
                # dist_avg = 0
                
                # avgs[0] = sum(errors[i])/buf_num
                # avgs[1] = sum(occupants[i])/buf_num
                # avgs[2] = sum(avg_dists[i])/buf_num
                if rollover == (num_cams):
                    rollover = 0
                    index = index + 1
                
                # for j in range(buf_num):
                #     err_avg = err_avg + errors[i][j]
                #     occupant_avg = occupant_avg + occupants[i][j]
                #     dist_avg = dist_avg + avg_dists[i][j]
                #     avgs[1] = 9 + j
                # avgs[0] = err_avg/buf_num
                # avgs[1] = occupant_avg/buf_num
                # avgs[2] = dist_avg/buf_num
                # a = a + 1
    except:
        avgs[0] = 'Error'
        avgs[4] =  str(sys.exc_info())

    return

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



def queue_parse(all_output_stats, buf_num, avgs, removed):
    #each entry within this will be a list of occupants, errors, avg dists
    all_out = [[[None]*3]* buf_num] * len(all_output_stats)
    a = 0
    time.sleep(10)
    try:
        while(True):
            #location of oldest entry in mega list
            index = a % buf_num
            
            #for every queue passed in, replace oldest entry in mega list and calculate avgs
            for i, q in enumerate(all_output_stats):
                #if the q has readable values, replace oldest entry with the new stats list
                # if not q.empty():
                all_out[i][index] = all_output_stats[i].get()
               
                    # if not q.full() and not removed.value:
                    #     q.put(recent[i])
                
                # reset avgs
                err_avg = 0
                occupant_avg = 0
                dist_avg = 0
                
                #cycle through every stats list in this queue to calculate avgs
                for j in range(buf_num):
                    err_avg = err_avg + all_out[i][j][0]
                    occupant_avg = occupant_avg + all_out[i][j][1]
                    # dist_avg = dist_avg + all_out[i][j][2]
                    avgs[1] = j
                    
                avgs[0] = 3
                avgs[0] = err_avg/buf_num
                avgs[1] = occupant_avg/buf_num
                # avgs[2] = dist_avg/buf_num
                
            #increment counter
            a = a + 1
    except:
        print("analysis error")
    return

    

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