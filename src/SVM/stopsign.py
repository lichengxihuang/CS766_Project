import os

import cv2
import numpy as np
import glob
from PIL import Image

def main():
    stop = cv2.imread('1.png')
    stop = cv2.resize(stop, (500, 500))
    cv2.imshow("origin",stop)
    # gray = cv2.cvtColor(stop, cv2.COLOR_BGR2GRAY)
    # # binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    #
    # # find contours in the thresholded image and initialize the
    # # shape detector
    #
    # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # for cnt in contours:
    #
    #     epsilon = 0.005*cv2.arcLength(cnt,True)
    #     approx = cv2.approxPolyDP(cnt,epsilon,True)
    #
    #     size = len(approx)
    #
    #     print(size)
    #     cv2.line(stop, tuple(approx[0][0]), tuple(approx[size-1][0]),(0,255,0),3)
    #     for k in range(size - 1):
    #         cv2.line(stop, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3)
    #     if size == 8:
    #         setLabel(stop, "o", cnt)
    # cv2.imshow("img",stop)


    # cv2.imshow("mask", gray)

    # preProcessing(gray)
    mask,res = colorseg(stop)



    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    cv2.waitKey()

def setLabel(img,str,contour):
    (text_width,text_height), baseline = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX,0.7,1)
    x,y,width,height = cv2.boundingRect(contour)
    pt_x = x+int((width-text_width)/2)
    pt_y = y + int((height + text_height) / 2)
    cv2.rectangle(img, (pt_x, pt_y + baseline),(pt_x + text_width, pt_y - text_height), (200,200,200), cv2.FILLED)
    cv2.putText(img,str,(pt_x,pt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, 8)



def colorseg(stop):
    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))
    hsv = cv2.cvtColor(stop, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    lower_red = np.array([20, 80, 50])
    upper_red = np.array([255, 255, 220])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(stop, stop, mask=mask)

    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    mask = maskClose

    conts, h = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(res, conts, -1, (255, 0, 0), 3)
    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        cv2.rectangle(res, (x, y), (x + w , y + h ), (0, 0, 255), 2)

        crop_img = res[y:y + h, x:x + w]

        crop_img = cv2.resize(crop_img, (64,128))
        hog = cv2.HOGDescriptor()

        h = hog.compute(crop_img)
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(0)
    return mask, res



main()