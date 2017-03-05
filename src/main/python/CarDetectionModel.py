import os
import numpy as np
from utils import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
import logging

def getFileNames(dir):
    fileNames = []
    for (dirName, dirs, files) in os.walk(dir):
        names = [os.path.join(dirName, file) for file in files]
        fileNames.extend(names)
    return fileNames

def generateDataset(vehiclesDir, nonVehiclesDir):
    vehicleFiles = getFileNames(vehiclesDir)
    logging.info("Total vehicles files: {}".format(len(vehicleFiles)))
    vehicleLabels = np.ones(len(vehicleFiles))
    nonVehicleFiles = getFileNames(nonVehiclesDir)
    logging.info("Total non-vehicles files: {}".format(len(nonVehicleFiles)))
    nonVehicleLabels = np.zeros(len(nonVehicleFiles))
    files = np.concatenate([vehicleFiles, nonVehicleFiles])
    labels = np.concatenate([vehicleLabels, nonVehicleLabels])
    return shuffle(files, labels)

def generateModel(files, labels, outputFile = None):
    color_space = 'YCrCb'
    spatial_size = (32, 32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'

    features = extract_features(files, color_space, spatial_size,
                     hist_bins, orient,
                     pix_per_cell, cell_per_block, hog_channel)
    logging.info("Feature extraction complete. Dataset size: {}".format(features.shape))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)

    logging.info("Features scaling complete.")

    svc = LinearSVC()

    pipeline = Pipeline([
        ("scaler", scaler),
        ("svc", svc)
    ])
    parameters = {'svc__C': np.logspace(-5,2, num=8)}

    clf = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1)
    logging.info("Fitting the model...")
    clf.fit(X_train, y_train)
    logging.info("Model fitting complete.")
    if outputFile is not None:
        estimator = clf.best_estimator_
        data = {
            "estimator": estimator,
            "color_space": color_space,
            "spatial_size": spatial_size,
            "hist_bins": hist_bins,
            "orient": orient,
            "pix_per_cell": pix_per_cell,
            "cell_per_block": cell_per_block,
            "hog_channel": hog_channel
        }
        with open(outputFile, "wb") as f:
            pickle.dump(data, f)

    best = clf.best_estimator_
    y_pred = best.predict(X_test)
    logging.info("C: {}".format(best.get_params()["svc"].C))
    logging.info("Classification performance:\n{}".format(classification_report(y_test, y_pred)))

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    files, labels = generateDataset("vehicles", "non-vehicles")
    generateModel(files, labels, "SVCModel-YCrCb.p")

