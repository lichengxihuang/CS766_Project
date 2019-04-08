#Generate positive training dataset

import os

import cv2
import numpy as np
import glob
from PIL import Image

def read_img():
    filenames = glob.glob("positive/*.jpg")

    images = [cv2.imread(img) for img in filenames]
    return images

def positive_process(images):
    i = 0
    for image in images:
        resized_img = cv2.resize(image, (int(500), int(500)))
        resized_img = colorseg(resized_img)

        img_name = 'resizeimg' + str(i) + '.jpg'

        path = '\output_neg'
        # cv2.imwrite(os.path.join(path, img_name), resized_img)
        cv2.imwrite(img_name, resized_img)
        cv2.waitKey(0)

        i += 1

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
    crop_img = stop
    crop_img_ori = stop
    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        cv2.rectangle(res, (x, y), (x + w , y + h ), (0, 0, 255), 2)

        crop_img = res[y:y + h, x:x + w]

        crop_img = cv2.resize(crop_img, (112,112))

        crop_img_ori = stop[y:y + h, x:x + w]

        crop_img_ori = cv2.resize(crop_img_ori, (112,112))

        # hog = cv2.HOGDescriptor()
        #
        # h = hog.compute(crop_img)

        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
    return crop_img_ori


def main():
    imgs = read_img()
    positive_process(imgs)

main()