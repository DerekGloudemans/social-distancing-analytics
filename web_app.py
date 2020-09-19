# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:53:52 2020

@author: Nikki
"""
#useful references:
#https://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent
#https://github.com/miguelgrinberg/Flask-SocketIO

from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context, Response
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import random
import datetime
import sys
import multiprocessing as mp
import multiprocess_video as mv
import analyze_data as adat
import time
from ctypes import c_bool
import cv2
import base64

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()


def background_thread():
    """Example of how to send server generated events to clients."""
    img = None
    while True:
        socketio.sleep(.5)
        if not image_q.empty():
            frame = image_q.get()
            wdt = frame.shape[1]
            hgt = frame.shape[0]
            scale= 419/wdt
            dim = (int(wdt * scale), int(hgt * scale))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            
            success, img = cv2.imencode('.jpg', frame)
            img = img.tobytes()
            img = base64.b64encode(img).decode('utf-8')
            fin_img = "data:image/jpeg;base64,{}".format(img)
                
            occupants = adat.total_o_avg(ocpts)
            errors = adat.total_e_avg(errs)
            avg_dist = adat.total_dist_avg(dists)
            if occupants == 0:
                compliance = 100
            else:
                compliance = round((1-(errors/occupants)), 4) * 100
            dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            socketio.emit('update',
                          {'occ': occupants, 'comp': compliance, 'dist': avg_dist, 'time': dt, 'image':fin_img},
                          namespace='/test')
        # else:
        #     socketio.emit('update',
        #                   {'occ': 'nan', 'comp': 'nan', 'dist': 'nan', 'time': 'nan'},
        #                   namespace='/test')

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

def tester(errs, ocpts, dists):
    count = 5
    while True:
        time.sleep(.5)
        count = count + 1
        errs.append(count)
        ocpts.append(count)
        dists.append(count)
    
def gen():
    while True:
        if not image_q.empty():
            frame = image_q.get()
            success, encodeImg = cv2.imencode('.jpg', frame)
            # if improperly encoded, retry
            if not success:
                continue
            
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodeImg) + b'\r\n')
    
@app.route('/vid_feed')
def vid_feed():
    if not image_q.empty():
        return Response(gen(), mimetype = 'multipart/x-mixed-replace; boundary=frame')
    
    
if __name__ == '__main__':
    
    try:
        # app.run(host='192.168.86.245', port=8000, debug=True,
        #                      use_reloader=False)
        manager = mp.Manager()
        buf_num = 3
        global errs
        global ocpts
        global dists
        errs = manager.list()
        ocpts = manager.list()
        dists = manager.list()
        
        #FIXME need a better way to do this (should be based on how many cameras initialize)
        #should initialize cameras here instead of in mp vid
        num_cams = 2

        updated = manager.Value(c_bool, False)
        frames = manager.list([None]* num_cams)
        times = manager.list([None]* num_cams)
        avgs = manager.list([None] * 5)
        avg_lock = manager.Lock()
        i_lock = manager.Lock()
        out_q = manager.Queue(num_cams*2)
        bbox_q = manager.Queue()
        ind = manager.Value(int, 0)
        
        global image_q
        image_q = manager.Queue(num_cams*2)
        
        for i in range(num_cams):
            errs.append(manager.list([None]))
            ocpts.append(manager.list([None]))
            dists.append(manager.list([None]))
        # var_list = manager.list([errs, ocpts, dists, updated, frames, times, avgs, avg_lock, i_lock, index]) 
        # errs = m.list([None]*buf_num)
        # ocpts = m.list([None]*buf_num)
        # dists = m.list([None]*buf_num)
        # proc = mp.Process(target=tester, args=(errs, ocpts, dists,))
        #might be good to make this a background task in socketio
        proc = mp.Process(target=mv.main, args=(errs, ocpts, dists, updated, frames, times, avgs, avg_lock, i_lock, ind, out_q, bbox_q, image_q, ))
        # thread = socketio.start_background_task(target = tester, args = (errs,))
        # proc.daemon = True
        proc.start()
        while bbox_q.empty():
            socketio.sleep(.5)
        socketio.run(app, host='169.254.45.230/16', port=8000, debug=True,
                              use_reloader=False)
        # while True:
        #     if not image_q.empty():
        #         result = image_q.get()
        #         cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        #         cv2.imshow("result", result)
        #         if cv2.waitKey(1) & 0xFF == ord('q'): break
        proc.terminate()
        # frame = image_q.get()
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        # cv2.waitkey(5)
        cv2.destroyAllWindows()
    except:
        print("Unexpected error:", sys.exc_info())
        proc.terminate()
        socketio.stop()
        # frame = image_q.get()
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        # cv2.waitkey(5)
        cv2.destroyAllWindows()
        
        
