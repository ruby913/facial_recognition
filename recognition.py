# This Python function take photos from the Webcam, detects and recognizes face from the machine learning model.
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
import time
import greengrasssdk
import platform
from threading import Timer
import json
from picamera import PiCamera
from picamera.array import PiRGBArray

client = greengrasssdk.client('iot-data')
#You can place the face which you want the model to recognize in this folder
my_faces_path = './positive_faces'
#You can put more face pics as training data set to improve the accurancy.
other_faces_path = './negative_faces'
model_path = './model'
size = 64
camera = PiCamera()
rawCapture = PiRGBArray(camera)


x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    W1 = weightVariable([3,3,3,32]) 
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob_5)

    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

output = cnnLayer()  
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint(model_path))  
   
def is_my_face(image):  
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
    if res[0] == 1:  
        return True  
    else:  
        return False  

detector = dlib.get_frontal_face_detector()

# cam = cv2.VideoCapture(0)  
   
while True:  
    camera.capture('myface.jpg');
    _, img = cv2.imread('myface.jpg')  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff  
        if key == 27:
            sys.exit(0)
            
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]

        face = cv2.resize(face, (size,size))
        
	record_data = {}
	record_data['result'] = is_my_face(face)

	prepared_record_data = json.dumps(record_data)

	client.publish(topic = 'record/webcam', payload = prepared_record_data)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)
  
sess.close()
