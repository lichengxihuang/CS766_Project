import cv2
import numpy as np
import glob

SIZE = 32

def load_training_data():
    dataset = []
    labels = []
    pos_filenames = glob.glob("output_pos/*.jpg")
    for img in pos_filenames:
        image = cv2.imread(img, 0)
        img = cv2.resize(image, (SIZE, SIZE))
        # img = np.reshape(img, [SIZE, SIZE])
        dataset.append(img)
        labels.append(1)

    neg_filenames = glob.glob("output_neg/*.jpg")
    for img in neg_filenames:
        image = cv2.imread(img,0)
        img = cv2.resize(image, (SIZE, SIZE))
        # img = np.reshape(img, [SIZE, SIZE])
        dataset.append(img)
        labels.append(0)

    return np.array(dataset), np.array(labels)



def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SIZE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img




def get_hog() :
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
                            cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,
                            nlevels, signedGradient)

    return hog



class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)



class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def traning():
    print('Loading training data from output_pos and output_neg')
    # Load data.

    data, labels = load_training_data()
    print(data.shape)
    print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(0)
    shuffle = rand.permutation(len(data))
    data, labels = data[shuffle], labels[shuffle]

    print('Deskew images ... ')
    data_deskewed = list(map(deskew, data))

    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog()

    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    for img in data_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print('Training SVM model ...')
    model = SVM()
    model.train(hog_descriptors, labels)

    print('Saving SVM model ...')
    model.save('data_svm.dat')
    return model


def getLabel(model, data):
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = [cv2.resize(gray,(SIZE,SIZE))]
    #print(np.array(img).shape)
    img_deskewed = list(map(deskew, img))
    hog = get_hog()
    hog_descriptors = np.array([hog.compute(img_deskewed[0])])
    hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
    return int(model.predict(hog_descriptors)[0])