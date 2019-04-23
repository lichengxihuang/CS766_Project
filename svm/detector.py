import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from skimage.feature import hog
from img_processing import get_rois
import os
import glob
import pickle
import argparse




def read_img(path):
    imgs = []
    for filename in os.listdir(path):
        full_filename = path + '/' + filename
        print(full_filename)
        if os.path.isdir(full_filename) or filename.startswith('.'):
            continue
        imgs.append(cv2.imread(full_filename))

    return imgs




def extract_hog(imgs):
    res = [hog(img) for img in imgs]
    return np.array(res)


def train():
    train_pos_path = 'dataset/train/pos'
    train_neg_path = 'dataset/train/neg'

    pos_imgs = read_img(train_pos_path)
    neg_imgs = read_img(train_neg_path)
    X_pos = extract_hog(pos_imgs)
    X_neg = extract_hog(neg_imgs)

    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))

    X, y = shuffle(X, y, random_state=0)

    clf = SVC(gamma='scale', class_weight={1: len(neg_imgs)/len(pos_imgs)})
    clf.fit(X, y)

    return clf


def test(clf):
    testfilename = 'WX20190423-102222@2x.png'
    in_dir = 'test'
    out_dir = 'test'

    # in_dir = 'original/positive'
    # out_dir = 'results/original_positive'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for filename in os.listdir(in_dir):
        full_filename = in_dir + '/' + filename
        if os.path.isdir(full_filename) or filename.startswith('.'):
            continue
        img = cv2.imread(full_filename)
        rois, points = get_rois(img, ret_points=True)
        if not points:
            continue

        X = extract_hog(rois)
        preds = clf.predict(X)

        print(preds)

        for i in range(len(preds)):
            if preds[i]:
                point = points[i]
                cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 2)
        cv2.imwrite(out_dir + '/' + 'out_' + filename, img)




    # in_dir = 'original/videos'
    # out_dir = 'results/videos'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #
    # for filename in glob.glob(in_dir + '/' + '*.mp4'):
    #     cap = cv2.VideoCapture(filename)
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     # out = cv2.VideoWriter(out_dir + '/result_' + os.path.basename(filename), 0x7634706d, 30, (int(cap.get(3)), int(cap.get(4))))
    #     out = cv2.VideoWriter(out_dir + '/result_' + os.path.basename(filename), fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
    #
    #     while (cap.isOpened()):
    #
    #         ret, img = cap.read()
    #
    #         if ret == True:
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #
    #             rois, points = get_rois(img, ret_points=True)
    #
    #             if rois:                        # only predict when there are rois
    #                 X = extract_hog(rois)
    #                 preds = clf.predict(X)
    #
    #                 # print(preds)
    #
    #                 for i in range(len(preds)):
    #                     if preds[i]:
    #                         point = points[i]
    #                         cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 2)
    #
    #             out.write(img)
    #
    #         else:
    #             break
    #
    #
    #     cap.release()
    #     out.release()
    #     cv2.destroyAllWindows()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('m', type=int, help='0:train, 1:test')

    args = parser.parse_args()

    if args.m == 0:
        clf = train()
        with open('svm_clf.pkl', 'wb') as f:
            pickle.dump(clf, f)

    elif args.m == 1:
        with open('svm_clf.pkl', 'rb') as f:
            clf = pickle.load(f)
        test(clf)

    else:
        parser.error("Can only be 0 or 1.")





