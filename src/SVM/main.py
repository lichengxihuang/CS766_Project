import cv2
import numpy as np
from math import sqrt
import imutils
import preprocess_frame as preprocess
import SVM



def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    img2 = np.zeros((labels.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    t = 0
    for i in range(0, nb_components - 1):
        if sizes[i] >= threshold:
            img2[labels == i + 1] = 255
            t += 1
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2(or_better=True) else cnts[1]
    return cnts

# def contourIsSign(perimeter, centroid, threshold):
#     #  perimeter, centroid, threshold
#     # # Compute signature of contour
#     result=[]
#     for p in perimeter:
#         p = p[0]
#         distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
#         result.append(distance)
#     max_value = max(result)
#     signature = [float(dist) / max_value for dist in result]
#     # Check signature of contour.
#     temp = sum((1 - s) for s in signature)
#     temp = temp / len(signature)
#     if temp < threshold: # is  the sign
#         return True, max_value + 2
#     else:                 # is not the sign
#         return False, max_value + 2

#crop sign
# def cropContour(image, center, max_distance):
#     width = image.shape[1]
#     height = image.shape[0]
#     top = max([int(center[0] - max_distance), 0])
#     bottom = min([int(center[0] + max_distance + 1), height-1])
#     left = max([int(center[1] - max_distance), 0])
#     right = min([int(center[1] + max_distance+1), width-1])
#     print(left, right, top, bottom)
#     return image[left:right, top:bottom]

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]


def findLargestSign(image, contours, threshold, distance_theshold,model):
    max_distance = 0
    coordinate = None
    sign = None
    havesign = None
    for c in contours:
        # M = cv2.moments(c)
        # if M["m00"] == 0:
        #     continue
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        # is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        # if is_sign and distance > max_distance and distance > distance_theshold:
        #     max_distance = distance
        #     coordinate = np.reshape(c, [-1,2])
        #     left, top = np.amin(coordinate, axis=0)
        #     right, bottom = np.amax(coordinate, axis = 0)
        #     coordinate = [(left-2,top-2),(right+3,bottom+1)]
        #     sign = cropSign(image,coordinate)
        #     havesign = True
        #####################################
        coordinate = np.reshape(c, [-1, 2])
        left, top = np.amin(coordinate, axis=0)
        right, bottom = np.amax(coordinate, axis = 0)
        coordinate = [(left-2,top-2),(right+3,bottom+1)]
        sign = cropSign(image,coordinate)
        sign_type = SVM.getLabel(model, sign)
        if sign_type == 1:
            break
        ####################################

    return sign, coordinate


# def findSigns(image, contours, threshold, distance_theshold):
#     signs = []
#     coordinates = []
#     for c in contours:
#         # compute the center of the contour
#         M = cv2.moments(c)
#         if M["m00"] == 0:
#             continue
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
#         if is_sign and max_distance > distance_theshold:
#             sign = cropContour(image, [cX, cY], max_distance)
#             signs.append(sign)
#             coordinate = np.reshape(c, [-1,2])
#             top, left = np.amin(coordinate, axis=0)
#             right, bottom = np.amax(coordinate, axis = 0)
#             coordinates.append([(top-2,left-2),(right+1,bottom+1)])
#     return signs, coordinates


def localization(image, min_size_components, similitary_contour_with_circle,
                 model, count):
    original_image = image.copy()
    binary_image = preprocess.preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

    binary_image = cv2.bitwise_and(binary_image, binary_image, mask = remove_other_color(image))

    contours = findContour(binary_image)

    sign, coordinate = findLargestSign(original_image, contours,
                                                 similitary_contour_with_circle, 15,model)

    text = "stop sign"
    sign_type = -1

    sign_type = SVM.getLabel(model, sign)
    cv2.imwrite(str(count) + '_' + text + '.png', sign)
    # if havesign:
    #     sign_type = SVM.getLabel(model, sign)
    #     cv2.imwrite(str(count) + '_' + text + '.png', sign)

    if sign_type > 0: #and sign_type != current_sign_type:
        cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2,
                    cv2.LINE_4)
    return coordinate, original_image, sign_type, text

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # define range of white color in HSV
    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # define range of black color in HSV
    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # define range of red color in HSV
    lower_red = np.array([20, 80, 50], dtype=np.uint8)
    upper_red = np.array([255, 255, 220], dtype=np.uint8)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    #remove other colors and select only red
    mask_1 = cv2.bitwise_not(mask_blue, mask_white, mask_black)
    # mask = cv2.bitwise_and(mask_1, mask_red)

    mask = mask_1
    # mask = mask_red
    return mask

def main():
    model = SVM.traning()
    similitary_contour_with_circle = 0.65
    min_size_components = 1000

    test_frame = cv2.imread('1.png')
    count = 0

    frame = cv2.resize(test_frame, (640, 480))

    coordinate, image, sign_type, text = localization(frame, min_size_components,
                                                      similitary_contour_with_circle,
                                                      model, count)

    cv2.imshow('result for test image',image)
    cv2.waitKey(0)
    return 0

main()