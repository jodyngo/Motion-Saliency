# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 07:04:19 2018

@author: Mason Proco
"""

import cv2
cap = cv2.VideoCapture(0)
saliency = None

while True:
    ret, frame = cap.read()
    
    if saliency is None:
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        saliency.init()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    cv2.imshow("Map", saliencyMap)
    
    
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()
