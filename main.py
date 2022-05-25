from mhi import generate_mhi
from feature_desciptors import generate_sift, generate_orb
from glob import glob
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline

video_class_paths = glob("database/*")
# video_class_paths = rnd.sample(video_class_paths, 4)
# video_class_paths = ["database/HighJump", "database/YoYo",
#  "database/PullUps", "database/PushUps"]
class_labels = [os.path.split(path)[-1]
                for path in video_class_paths]
print(f"Labels: {class_labels}")

dictionary_size = len(class_labels)

descs = []
labels = []

bow = cv.BOWKMeansTrainer(dictionary_size)


def pad(array, reference_shape, offsets):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                  for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


def main(*, generate_mhi_imgs=False, feature_detector='sift',
         generate_imgs=False, model='svm', cross_val=False) -> None:
    for idx, class_path in enumerate(video_class_paths):
        print(f"Preprocessing {os.path.split(class_path)[-1]} videos")
        for path in glob(f"{class_path}/*.avi"):
            if generate_mhi_imgs:
                # generate_mhi(path, class_path, every_n_frames=25)
                generate_mhi(path, class_path)
        for path in glob(f"{class_path}/*_mhi.png"):
            if feature_detector == 'sift':
                des = generate_sift(path,
                                    class_path if generate_imgs else None)
                if des is not None:
                    bow.add(des)
            if feature_detector == 'orb':
                des = generate_orb(path, class_path if generate_imgs else None)
                # des = np.array(new_des, dtype=np.float32)
                if des is not None:
                    des = pad(des, (384, 32), [0, 0])
                    descs.append(des.flatten())
                    labels.append(idx)

    if feature_detector == 'sift':
        dictionary = bow.cluster()
        flann = cv.FlannBasedMatcher(
            dict(algorithm=0, trees=5), dict(checks=50))
        extractor = cv.SIFT_create()
        bowDict = cv.BOWImgDescriptorExtractor(extractor, flann)
        bowDict.setVocabulary(dictionary)

        for idx, class_path in enumerate(video_class_paths):
            print(f"Feature extracting from {os.path.split(class_path)[-1]}")
            for path in glob(f"{class_path}/*_mhi.png"):
                img = cv.imread(path)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                if feature_detector == 'sift':
                    sift = cv.SIFT_create()
                    desc = bowDict.compute(gray, sift.detect(gray))
                    if desc is not None:
                        descs.extend(desc)
                        labels.append(idx)

    X = np.array(descs)
    y = np.array(labels)

    # X = descs
    # y = labels

    if model == 'svm':
        clf = make_pipeline(
            StandardScaler(),
            SVC(random_state=1, kernel="poly", coef0=7.9)
        )
    if model == 'nn':
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(solver='lbfgs', alpha=.001,
                          hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000, activation='relu')
        )

    if cross_val:
        folds = 10
        cross_validator = KFold(n_splits=folds, random_state=1, shuffle=True)

        scores = cross_val_score(
            clf, X, y, cv=cross_validator)

        print(
            f"K folds cross validations with {folds} folds: {scores.mean() * 100:.3f}%")
        print(f"Standard deviation: {scores.std()}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy * 100:.3f}%")

        confusion = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=class_labels)
        disp.plot(xticks_rotation="vertical")
        plt.show()


if __name__ == "__main__":
    main(generate_mhi_imgs=True, generate_imgs=True, model='nn')
