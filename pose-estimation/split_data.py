import os
from pathlib import Path
import random
import shutil
import json


if __name__ == "__main__":
    # folder that contains the st-gcn files (one per video)
    stgcn_json_path =  '../data/poses/normalized_st-gcn_format/'
    training_path =  '../data/st-gcn/training/'
    test_path = '../data/st-gcn/test/'

    train_labels_path = '../data/st-gcn/train_labels.json'
    test_labels_path = '../data/st-gcn/test_labels.json'
    train_labels, test_labels = {}, {}
    classes = data = json.load(open('../data/classes.json'))

    labels = {
        0: 'no_gestures',
        1:'gestures'
        }

    # cleaning previous split
    shutil.rmtree(training_path)
    os.makedirs(training_path)
    shutil.rmtree(test_path)
    os.makedirs(test_path)

    train_ratio = 0.6

    gesture_dataset = [stgcn_json_path + f for f in os.listdir(stgcn_json_path) if classes[f.rstrip('.json')]]
    no_gesture_dataset = [stgcn_json_path + f for f in os.listdir(stgcn_json_path) if not classes[f.rstrip('.json')]]

    n = min(len(gesture_dataset), len(no_gesture_dataset))
    n_train = round(train_ratio * n)

    print(len(gesture_dataset), len(no_gesture_dataset))

    gesture_dataset = random.sample(gesture_dataset, len(gesture_dataset))
    no_gesture_dataset = random.sample(no_gesture_dataset, len(no_gesture_dataset))
    train_set = gesture_dataset[:n_train] + no_gesture_dataset[:n_train]
    test_set = gesture_dataset[n_train:n] + no_gesture_dataset[n_train:n]   

    for path in train_set:
        file_name = path.split('/')[-1]
        pedestrian_id = file_name.split('.')[0]
        c = classes[pedestrian_id]
        train_labels[pedestrian_id] = {
            "has_skeleton": True,
            "label": labels[c],
            "label_index": c
            }
        dest_path = training_path + file_name
        # print('Copying {} to {}'.format(path, dest_path))
        shutil.copyfile(path, dest_path)
    with open(train_labels_path, 'w') as f:
        json.dump(train_labels, f)

    for path in test_set:
        file_name = path.split('/')[-1]
        pedestrian_id = file_name.split('.')[0]
        c = classes[pedestrian_id]
        test_labels[pedestrian_id] = {
            "has_skeleton": True,
            "label": labels[c],
            "label_index": c
            }
        dest_path = test_path + file_name
        # print('Copying {} to {}'.format(path, dest_path))
        shutil.copyfile(path, dest_path)
    with open(test_labels_path, 'w') as f:
        json.dump(test_labels, f)

            