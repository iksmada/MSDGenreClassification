#  use data/MSD_genre/msd_genre_dataset.txt as features
# !/usr/bin/python3 -W ignore::DeprecationWarning
#  -*- coding: UTF-8 -*-

import argparse
from time import time

import numpy as np
from sklearn.cluster import KMeans


def kmeans_reclass(X, y):
    n_classes = len(np.unique(y))
    model = KMeans(n_clusters=n_classes, init='k-means++', max_iter=500, n_init=10, n_jobs=-1)
    y_pred = model.fit_predict(X)

    # separate clusters
    clusters = dict()
    i = 0
    for clazz in y_pred:
        if clazz not in clusters:
            clusters[clazz] = []
        clusters[clazz].append(y[i])
        i = i + 1

    conversor = dict()
    for clazz, cluster in clusters.items():
        values, counts = np.unique(cluster, return_counts=True)
        conversor[clazz] = values[np.argmax(counts)]

    y_reclass = []
    for clazz in y_pred:
        y_reclass.append(conversor[clazz])

    return np.array(y_reclass)


if __name__ == '__main__':
    start_time = time()

    parser = argparse.ArgumentParser(description='Million Song Dataset Genre Classification')
    parser.add_argument('-i', '--input', type=str, help='Input data path',
                        #default="data/MSD_genre/msd_genre_dataset.csv")
                        default="extractedLetters/all_letters.csv")
    parser.add_argument('-s', '--size', type=float, help='Train size in % relative to test set',
                        default=0.8)
    parser.add_argument('-t', '--tree', type=int, help='Number of tree')
    parser.add_argument('-n', '--features', type=int, help='Number of features from the beginning')

    print("--- %s seconds ---" % (time() - start_time))
