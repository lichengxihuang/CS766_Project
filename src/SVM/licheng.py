import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



def main():

    # dir = '../dataset/'


    # for filename in os.listdir(dir):
    #     img = cv2.imread(dir + filename)
    #
    #     res = img_processing(img)
    #
    #     # cv2.imshow("Mask", res)
    #     # cv2.waitKey(0)
    #
    #     cv2.imwrite('open' + filename, res)


    # filename = os.listdir(dir)[5]

    filename = '1.png'
    img = cv2.imread(filename)

    res = img_processing(img)

    cv2.imwrite('open' + filename, res)




def img_processing(img):
    mask = get_mask(img)
    cv2.imshow("mask",mask)

    cv2.waitKey()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[2]
    # cnt = contours
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img, approx, 0, (0, 255, 0), 3)
    cv2.imshow("im2", img)
    cv2.waitKey(0)

    # mask = fill_holes(mask)
    # find_connected(mask)

    res = apply_mask(mask, img)

    return res



def find_connected(mask):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)



    print()



def get_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))

    mask = mask1 | mask2
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask


def fill_holes(mask):
    h, w = mask.shape[:2]

    mask_floodfill = mask.copy()
    cv2.floodFill(mask_floodfill, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)

    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)

    new_mask = mask | mask_floodfill_inv

    return new_mask
    # cv2.imshow("mask", mask)
    # cv2.imshow("new_mask", new_mask)
    # cv2.imshow("mask_floodfill", mask_floodfill)
    # cv2.imshow("mask_floodfill_inv", mask_floodfill_inv)
    # cv2.waitKey(0)




def apply_mask(mask, img):
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

    # sure_bg = cv2.dilate(opening, kernel, iterations=3)
    #
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #
    # ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    #
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    #
    # cv2.imshow("sure_bg", sure_bg)
    # cv2.imshow("sure_fg", sure_fg)
    # cv2.imshow("unknown", unknown)
    # cv2.waitKey(0)





if __name__ == '__main__':
    main()