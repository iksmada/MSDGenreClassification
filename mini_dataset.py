#  use data/MSD_genre/msd_genre_dataset.txt as features
# !/usr/bin/python3
#  -*- coding: UTF-8 -*-

import argparse
import sys
import os
import csv
from time import time



if __name__ == '__main__':
    start_time = time()

    parser = argparse.ArgumentParser(description='Million Song Dataset Genre Classification')
    parser.add_argument('-i', '--input', type=str, help='Input data path',
                        default="data/MSD_genre/msd_genre_dataset.txt")

    args = vars(parser.parse_args())
    print(args)
    INPUT = args["input"]

    X =[]
    fieldnames = []
    with open(INPUT, newline='') as csvfile:
        csvreader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        fieldnames = next(csvreader)
        print(fieldnames)
        for row in csvreader:
            X.append(row)

    print("--- %s seconds ---" % (time() - start_time))
