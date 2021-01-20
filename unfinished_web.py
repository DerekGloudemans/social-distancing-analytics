# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 23:06:03 2020

@author: nicol
"""

#useful references:
#https://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent
#https://github.com/miguelgrinberg/Flask-SocketIO

from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context, Response, url_for, redirect
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import random
import datetime
import sys
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
#FIXME make this actually secure???
app.config['SECRET_KEY'] = '403qrwebiup98hsan89-0-j2ojbeqfw08asdmnl23ir'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

#will want to pass the id of the page into the get methods for retrieving data


def background_thread():
    """Example of how to send server generated events to clients."""
    # img = None
    while True:
        socketio.sleep(5)

        occupants = random.random()
        compliance = random.random()
        avg_dist = random.random()
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fin_img = None
        socketio.emit('update',
                      {'occ': occupants, 'comp': compliance, 'dist': avg_dist, 'time': dt, 'image':fin_img},
                      namespace='/test')


@app.route('/', methods=['GET'])
def index():
    locations = ['All', 'Rand', 'MRB3', 'Commons']
    return render_template('fancy_index.html', async_mode=socketio.async_mode , title = "All", locations = locations)



@app.route('/<id1>', methods=['GET', 'POST'])
def cam(id1):
    titles = id1
    locations = ['All', 'Rand', 'MRB3', 'Commons']
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('fancy_index.html', async_mode=socketio.async_mode , title = titles, locations = locations)

@app.route('/downloads', methods=['GET', 'POST'])
def downloads():
    return render_template('downloads.html', async_mode=socketio.async_mode)



@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)


if __name__ == '__main__':
    try:
        # app.run(host='192.168.86.245', port=8000, debug=True,
        #                      use_reloader=False)

        # global errs
        # global ocpts
        # global dists

        # '10.66.46.173/16'
        socketio.run(app, host='127.0.0.1', port=8000, debug=True,
                              use_reloader=False)


        cv2.destroyAllWindows()
    except:
        print("Unexpected error:", sys.exc_info())
        socketio.stop()

        cv2.destroyAllWindows()