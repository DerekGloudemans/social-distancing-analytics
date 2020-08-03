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

def main(vids):
    buf_storage = [None] * len(vids)
    wait(10)
    while(True):
        
        for i, vid in enumerate(vids):
            while not vid.out_q.empty():
                buf_storage[i] = vid.out_q.get()
        
        
        
        for stat_buf in buf_storage:
            err_avg = sum(my_buf[0])/ len(my_buf[0])
            occupant_avg = sum(my_buf[1])/ len(my_buf[1])
            dist_avg = sum(my_buf[2])/ len(my_buf[2])
            print(err_avg)
            print(occupant_avg)
            print(dist_avg)
    

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