# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:45:46 2020

@author: nicol
"""
import threading as mp
import time
global l
def main():
    l = []
    streamer = mp.Thread(target=stream_all, args=(l,))
    streamer.daemon = True
    streamer.start()
    streamer.join()
    print(l)
    

def stream_all(l):
    l.append("hi")
    return

main()
