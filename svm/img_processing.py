import cv2
import numpy as np
import os

def main(cropped=False):
    # # in_dir = 'positive'
    # # out_dir = 'training/pos'
    # in_dir = 'original/new'
    # out_dir = 'result1_1'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # for filename in os.listdir(in_dir):
    #     print(filename)
    #
    #     full_filename = in_dir+'/'+filename
    #
    #     if os.path.isdir(full_filename) or filename.startswith('.'):
    #         continue
    #
    #     filename_list = filename.split('.')
    #     assert len(filename_list) == 2
    #
    #     img = cv2.imread(full_filename)
    #
    #     if cropped:
    #         rois = get_rois(img)
    #         for i, roi in enumerate(rois):
    #             cv2.imwrite(out_dir + '/sub_' + filename_list[0] + '_' + str(i) + '.' + filename_list[1], roi)
    #     else:
    #         rois, points = get_rois(img, ret_points=True)
    #         for i, point in enumerate(points):
    #             cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 2)
    #         cv2.imwrite(out_dir + '/out_' + filename, img)




    filename = 'img.jpg'

    filename_list = filename.split('.')

    img = cv2.imread(filename)

    rois, points = get_rois(img, ret_points=True)

    for i, roi in enumerate(rois):
        cv2.imwrite('sub_' + filename_list[0] + '_' + str(i) + '.' + filename_list[1], roi)

    # for i, point in enumerate(points):
    #     cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 2)
    # cv2.imwrite('out_' + filename, img)




def get_rois(img, ret_points=False):
    # scale to approximately size of 1280*720
    scale = 2 / (img.shape[0] / 720 + img.shape[1] / 1280)

    img = cv2.resize(img, None, fx=scale, fy=scale)

    # get_edges(img)
    mask = get_mask(img)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()


    mask = fill_holes(mask)

    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    mask = find_connected(mask, thresh=200)

    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cnt = contours[2]
    # hull = cv2.convexHull(cnt)
    # cv2.drawContours(img, [hull], 0, (0, 255, 0), 3)

    c = 1.1  # scale factor
    points = []  # store the points of bounding rectangles
    for cnt in contours:
        is_child = False
        x, y, w, h = cv2.boundingRect(cnt)
        c_x = x + w/2
        c_y = y + h/2

        x_1 = max(int(c_x - c * w / 2), 0)
        y_1 = max(int(c_y - c * h / 2), 0)
        x_2 = min(int(c_x + c * w / 2), img.shape[1] - 1)
        y_2 = min(int(c_y + c * h / 2), img.shape[0] - 1)

        curr = [x_1, y_1, x_2, y_2]

        # remove small bounding rectangles that are covered by some larger ones
        i = 0
        while i < len(points):
            point = points[i]
            if point[0] <= curr[0] and point[1] <= curr[1] and point[2] >= curr[2] and point[3] >= curr[3]:
                is_child = True
                break
            elif point[0] >= curr[0] and point[1] >= curr[1] and point[2] <= curr[2] and point[3] <= curr[3]:
                del points[i]
                i -= 1
            i += 1

        if not is_child:
            points.append(curr)

    # for point in points:
    #     cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 2)
    # res = img


    rois = []
    for point in points:
        x_1, y_1, x_2, y_2 = point
        curr = img[y_1:y_2 + 1, x_1:x_2 + 1]
        roi = cv2.resize(curr, (100, 100))
        rois.append(roi)

    if ret_points:
        points = [[int(a/scale) for a in point] for point in points]
        return rois, points
    else:
        return rois


# not used
def get_edges(img):

    ref_contour = np.array([[[295, 61]], [[61, 295]], [[61, 626]], [[295, 860]], [[626, 860]], [[860, 626]], [[860, 295]], [[626, 61]]])

    mask = cv2.Canny(img, 100, 200)
    mask = find_connected(mask, thresh=100)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    cv2.imshow("mask", img)
    cv2.waitKey()


    # mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(mask, 127, 255, 0)


    # h, w = mask.shape
    # cv2.floodFill(mask, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    # cv2.floodFill(mask, np.zeros((h + 2, w + 2), np.uint8), (0, h-1), 255)
    # cv2.floodFill(mask, np.zeros((h + 2, w + 2), np.uint8), (w-1, 0), 255)
    # cv2.floodFill(mask, np.zeros((h + 2, w + 2), np.uint8), (w-1, h-1), 255)
    #
    # cv2.imshow("mask", mask)
    # cv2.waitKey()
    # mask = cv2.bitwise_not(mask)
    # mask = fill_holes(mask)

    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    # mask = find_connected(mask, thresh=500)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    for cnt in contours:
        print(cv2.matchShapes(cnt, ref_contour, method=cv2.CONTOURS_MATCH_I1, parameter=0))

    cv2.imshow("mask", img)
    cv2.waitKey()


# find connected components and remove small ones
def find_connected(mask, thresh=0):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    for i in range(1, num_labels):
        if stats[i, -1] < thresh:
            mask[labels == i] = 0

    return mask



def get_mask(img):

    # convert to HSV
    # img = cv2.resize(img, (400, 425))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # select red color
    # mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    # mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))

    # mask1 = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([10, 255, 255]))
    # mask2 = cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255]))

    mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([150, 50, 40]), np.array([180, 255, 255]))


    mask = mask1 | mask2

    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    kernel = np.ones((3, 3), np.uint8)

    # dilation followed by erosion
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # erosion followed by dilation
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def fill_holes(mask):
    h, w = mask.shape

    mask_floodfill_1 = mask.copy()
    mask_floodfill_2 = mask.copy()
    mask_floodfill_3 = mask.copy()
    mask_floodfill_4 = mask.copy()

    # floodfill starting from four corner points
    cv2.floodFill(mask_floodfill_1, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    cv2.floodFill(mask_floodfill_2, np.zeros((h + 2, w + 2), np.uint8), (0, h-1), 255)
    cv2.floodFill(mask_floodfill_3, np.zeros((h + 2, w + 2), np.uint8), (w-1, 0), 255)
    cv2.floodFill(mask_floodfill_4, np.zeros((h + 2, w + 2), np.uint8), (w-1, h-1), 255)

    # bitwise or
    mask_floodfill = mask_floodfill_1 | mask_floodfill_2 | mask_floodfill_3 | mask_floodfill_4

    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)

    new_mask = mask | mask_floodfill_inv

    # cv2.imshow("mask", mask)
    # cv2.imshow("new_mask", new_mask)
    # cv2.imshow("mask_floodfill", mask_floodfill)
    # cv2.imshow("mask_floodfill_inv", mask_floodfill_inv)
    # cv2.waitKey()
    return new_mask




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
    main(cropped=True)
