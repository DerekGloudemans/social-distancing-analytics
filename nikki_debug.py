# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:46:35 2020

@author: Nikki
"""


import numpy as np
import tensorflow as tf
#from tensorflow import keras
from core.config import cfg
from core import utils
import cv2
from PIL import Image
import sys
from threading import Thread
from queue import Queue
import time

# tf.debugging.set_log_device_placement(True)

'''testing gpu assignment'''
# if tf.test.gpu_device_name():
#     print ('Device name: {}'.format(tf.test.gpu_device_name()))
#     print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# else:
#     print ('uh oh')

# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
#     print(c)

'''testing saving and loading a model'''
# inputs = tf.keras.Input(shape=(3,))
# x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
# outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# config1 = model.save_weights('yeahhh')
# print('stage1')
# new_model = tf.keras.Model.load_weights('config1')
# print('stage2')
# # model.save('my_model')

# # reconstructed_model = keras.models.load_model('my_model')

'''testing image resizing'''
# #probably change to uppercase
# INPUT_SIZE = 608

# 'read image, convert to pillow colorspace, and find shape'
# original_image = cv2.imread('./data/kite.jpg')
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# original_image_size = original_image.shape[:2]

# 'resize image and add another dimension'
# image = np.copy(original_image)
# target_size = [INPUT_SIZE, INPUT_SIZE]
# ih, iw    = target_size
# h,  w, _  = image.shape

# scale = min(iw/w, ih/h)
# nw, nh  = int(scale * w), int(scale * h)
# image_resized = cv2.resize(image, (nw, nh))

# 'creates new image of desired size'
# pad = ih//20

# image_resized = cv2.copyMakeBorder(image_resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])


# # image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
# # dw, dh = (iw - nw) // 2, (ih-nh) // 2
# # image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
# #image_paded = image_paded / 255.

# # image_paded = cv2.cvtColor(image_paded, cv2.COLOR_RGB2BGR())
# # cv2.imshow("result", image_paded)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# image = Image.fromarray(image_resized)
# image.show()

# '''testing multi-threading'''





# class VidThread():
    
#     def __init__(self, path, qSize=8):
#         self.vid = cv2.VideoCapture(path)
#         self.buff = Queue(maxsize=8)
#         self.stopped = False
        
#     def start(self):
#         t = Thread(target=self.update , args=())
#         t.daemon = True
#         t.start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped:
#                 return
#             if not self.buff.full():
#                 grab, frame = self.vid.read()
#                 if not grab:
#                     self.stop()
#                     return
#                 self.buff.put(frame)
                
#     def read(self):
#         return self.buff.get()
#     def stop(self):
#         self.stopped = True
#     def more(self):
#         return self.buff.qsize()>0
              
    
# # def main():
# print('starting')
# video_path = './data/road.mp4'

# thread = VidThread(video_path)

# thread.start()
# time.sleep(1.0)
# while thread.more():
#     print('reading')
#     cv2.namedWindow("result", cv2.WINDOW_NORMAL)
#     frame = thread.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(frame)
#     image = np.asarray(image)
#     result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.imshow("result", result)
#     cv2.waitKey(1)
        #thread.update()
# if __name__ == "__main__":
#    main()

# class VidThread():
    
#     def __init__(self, path, qSize=8):
#         self.vid = cv2.VideoCapture(path)
#         self.buff = Queue(maxsize=8)
#         self.stopped = False
        
#     def start(self):
#         #t = Thread(target=self.update, args=())
#         t = tf.train.Coordinator.register_thread(Thread(target=self.update, args=()))
#         t.daemon = True
#         t.start()
        
#         return self

#     def update(self):
#         while True:
            
#             if self.stopped:
#                 return
#             if not self.buff.full():
#                 print('update')
#                 grab, frame = self.vid.read()
#                 if not grab:
#                     self.stop()
#                     return
#                 self.buff.put(frame)
                
#     def read(self):
#         return self.buff.get()
#     def stop(self):
#         self.stopped = True
#         self.vid.release()
#         cv2.destroyAllWindows()
#     def more(self):
#         return self.buff.qsize()>0  

### figuring out what a pixel's location is

# import cv2      # import the OpenCV library                     
# import numpy as np  # import the numpy library

# font = cv2.FONT_HERSHEY_SIMPLEX

# video_capture = cv2.VideoCapture('./data/20190422_153844_DA4A.mkv')

# def mouseHandler(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
#         cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

# cv2.namedWindow("result", cv2.WINDOW_NORMAL)
# cv2.setMouseCallback("result", mouseHandler)


# while(True):

#     # Capture frame-by-frame
#     _, frame = video_capture.read()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     cv2.imshow("result", frame)

# video_capture.release()
# cv2.destroyAllWindows() 


###calculating transformation matrix
# unitx = np.array([1, 0, 0])
# unity = np.array([0, 1, 0])
# unitz = np.array([0, 0, 1])

# a = np.array([36.148342, -86.799332, 0])
# b = np.array([36.148139, -86.799375, 0])
# c = np.array([36.148349, -86.799135, 0])

# ba = (b-a)
# ca = (c-a)

# n = np.cross(ba, ca)

# w = n / np.linalg.norm(n)
# print(n)
# print(np.linalg.norm(n))
# print(w)

# u = ba / np.linalg.norm(ba)

# v = np.cross(w, u)
# r = [u,v,w]
# #r = [a, b, c]
# print(u)
# print(v)
# print(w)
# ra = np.matrix(r)
# print(ra)

# rb = ra.transpose()

# final1 = u*rb

# print(np.squeeze(np.asarray(final1)))
# final2 = v*rb

# print(np.squeeze(np.asarray(final2)))
# final3 = w*rb

# print(np.squeeze(np.asarray(final3)))

# d = np.array([0, 1658, 1406])
# e = np.array([0, 2493, 1190])
# f = np.array([0, 492, 990])

# de = (e-d)
# df = (f-d)

# n2 = np.cross(de, df)


# w2 = n2 / np.linalg.norm(n2)
# print(n2)
# print(np.linalg.norm(n2))
# print(w2)

# u2 = de / np.linalg.norm(de)

# v2 = np.cross(w2, u2)
# r2 = [u2,v2,w2]
# print(u2)
# print(v2)
# print(w2)
# ra2 = np.matrix(r2)
# print(ra2)

# rb2 = ra2.transpose()

# # final2 = d*rb2

# # print(np.squeeze(np.asarray(final2)))

# final1 = final1*ra2

# print(np.squeeze(np.asarray(final1)))
# final2 = final2*ra2

# print(np.squeeze(np.asarray(final2)))
# final3 = final3*ra2

# print(np.squeeze(np.asarray(final3)))

import transform as tran
# a = np.array([36.144187, -86.799707])   #far left street pole
# b = np.array([36.143990, -86.799594])   #pole by bike sign
# c = np.array([36.143997, -86.800180])   #corner of sidewalk
# d = np.array([36.144203, -86.800149])   #right of sidewalk stripe closest to camera
  
# e = np.array([18, 1151])
# f = np.array([462, 210])
# g = np.array([3286, 749])
# h = np.array([2940, 2150])

# x = np.array([a,b,c,d])
# y = np.array([e,f,g,h])

# m = tran.get_best_transform(x, y)
m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

a = np.array([[23, 43], [23, 23]])
b = np.array([[23, 43]])
c = np.array([[]])
# print(b.shape)
# print(b.size)
# print(b.flatten())
# c = np.array([b])
# print(tran.transform_pt_array(a,m))

# print(a.size)
print(tran.transform_pt_array(c,m))








