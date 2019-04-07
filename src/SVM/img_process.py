import os

import cv2
import numpy as np
import glob
from PIL import Image

def read_img():
    filenames = glob.glob("negative/*.jpg")

    images = [cv2.imread(img) for img in filenames]
    return images


def negative_process(images):
    i = 0
    for image in images:
        resized_img = cv2.resize(image, (int(100), int(100)))
        # cv2.imshow('res', resized_img)

        cv2.waitKey()

        img_name = 'resizeimg' + str(i) + '.jpg'

        path = '\output_neg'
        # cv2.imwrite(os.path.join(path, img_name), resized_img)
        cv2.imwrite(img_name, resized_img)
        cv2.waitKey(0)

        i += 1

def main():
    imgs = read_img()
    negative_process(imgs)

main()