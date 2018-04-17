import os
from pathlib import Path
from os import listdir
import random
from shutil import copyfile


if __name__ == "__main__":
    # folder that contains the st-gcn files (one per video)
    stgcn_json_path =  '../data/poses/normalized_st-gcn_format/'
    training_path =  '../data/st-gcn/training/'
    test_path = '../data/st-gcn/test/'

    train_ratio = 0.6

    dataset = [stgcn_json_path + f for f in listdir(stgcn_json_path)]

    n = len(dataset)
    n_train = round(train_ratio * n)
    shuffled_set = random.sample(dataset, n)
    train_set = shuffled_set[:n_train]
    test_set = shuffled_set[n_train:]

    for path in train_set:
        file_name = path.split('/')[-1]
        dest_path = training_path + file_name
        print('Copying {} to {}'.format(path, dest_path))
        copyfile(path, dest_path)

    for path in test_set:
        file_name = path.split('/')[-1]
        dest_path = test_path + file_name
        print('Copying {} to {}'.format(path, dest_path))
        copyfile(path, dest_path)


            