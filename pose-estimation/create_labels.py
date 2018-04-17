import os
from pathlib import Path

if __name__ == "__main__":

    # file that will contain the labels for each video in the training set
    labels_file = '../data/st-gcn/train_label.json'

    # file that will contain the labels for each video in the test set
    labels_file = '../data/st-gcn/test_label.json'

    list_video_names(videos_path, video_list_file)

    p = Path(openpose_json_path)
    
    labels = {}
    with open(video_list_file, 'r') as f: