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

def main(all_output_stats, buf_num, avgs):

    #########FIXME not currently putting output stats into avgs properly
    while(True):
        for q in all_output_stats:
            # read stats into temp buffer
            recent = [None] * buf_num
            for i in recent:
                
                recent[i] = q.get_nowait()
                
                if not q.full():
                    avgs[0] = 1

                    q.put(recent[i])
                        
            err_avg = 0
            occupant_avg = 0
            dist_avg = 0
            
            for i in range(buf_num):
                avgs[0] = 2
                
                err_avg = err_avg + recent[i][0]
                occupant_avg = occupant_avg + recent[i][1]
                dist_avg = dist_avg + recent[i][2]
            avgs[0] = err_avg/len(recent)
            avgs[1] = occupant_avg/len(recent)
            avgs[2] = dist_avg/len(recent)
            
            
    #     for i, vid in enumerate(vids):
    #         while not vid.out_q.empty():
    #             buf_storage[i] = vid.out_q.get()
        
        
        
    #     for stat_buf in buf_storage:
    #         err_avg = sum(my_buf[0])/ len(my_buf[0])
    #         occupant_avg = sum(my_buf[1])/ len(my_buf[1])
    #         dist_avg = sum(my_buf[2])/ len(my_buf[2])
    #         print(err_avg)
    #         print(occupant_avg)
    #         print(dist_avg)
    

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