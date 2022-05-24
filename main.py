from mhi import generate_mhi
from feature_desciptors import generate_sift, generate_orb
from glob import glob
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

video_class_paths = glob("database/*")
class_labels = [os.path.split(path)[-1]
                for path in video_class_paths]

dictionary_size = len(class_labels)

descs = []
labels = []

bow = cv.BOWKMeansTrainer(dictionary_size)


def main() -> None:
    for class_path in video_class_paths:
        for path in glob(f"{class_path}/*.avi"):
            # mhi = generate_mhi(path, class_path)
            # des = generate_sift(f"{path[:-4]}_mhi.png", class_path)
            des = generate_sift(f"{path[:-4]}_mhi.png")
            # des = generate_orb(f"{path[:-4]}_mhi.png", class_path)
            # des = generate_orb(f"{path[:-4]}_mhi.png")
            if des is not None:
                bow.add(des)

    dictionary = bow.cluster()
    flann = cv.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    extractor = cv.SIFT_create()
    # extractor = cv.ORB_create()
    bowDict = cv.BOWImgDescriptorExtractor(extractor, flann)
    bowDict.setVocabulary(dictionary)

    for idx, class_path in enumerate(video_class_paths):
        for path in glob(f"{class_path}/*.avi"):
            img = cv.imread(f"{path[:-4]}_mhi.png")
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # return bowDict.compute(gray, cv.SIFT().detect(gray))
            sift = cv.SIFT_create()
            descs.extend(bowDict.compute(gray, sift.detect(gray)))
            # orb = cv.ORB_create()
            # descs.extend(bowDict.compute(gray, orb.detect(gray)))
            labels.append(idx)

    X = np.array(descs)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1)

    clf = SVC(random_state=2, kernel="poly", C=1, degree=3, coef0=7.9)
    # clf = MLPClassifier(solver='lbfgs', alpha=.001,
    #                     hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000, activation='relu')

    ################################
    ### k-folds cross validation ###
    ################################

    # folds = 10
    # cross_validator = KFold(n_splits=folds, random_state=1, shuffle=True)

    # # linear = 0.378
    # # rbf = 0.433
    # # poly = 0.439
    # # sigmoid = 0.256

    # scores = cross_val_score(
    #     clf, X, y, cv=cross_validator)

    # print(
    #     f"K folds cross validations with {folds} folds: {scores.mean() * 100:.3f}%")
    # print(f"Standard deviation: {scores.std()}")

    ########################
    ### Confusion matrix ###
    ########################

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
    main()
