import os
from pathlib import Path
import json
from pprint import pprint
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # folder that contains all the video that have been treated by OpenPose
    videos_path = '../data/videos'
    # folder that contains the st-gcn files (one per video)
    stgcn_json_path =  '../data/poses/st-gcn_format/'

    p = Path(stgcn_json_path)
    
    for path in p.glob('*.json'):
        json_path = str(path)
        data = json.load(open(json_path))
        frames = data['data']
        ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        for frame in ordered_frames:
            # print(frame['frame_index'])
            x_joints, y_joints = [], []
            skeletons = frame['skeleton']
            for skeleton in skeletons:
                coordinates = skeleton['pose']
                for i in range(0, len(coordinates), 2):
                    x_joints += [coordinates[i]]
                    y_joints += [-coordinates[i+1]]
            plt.scatter(x_joints, y_joints)
            plt.draw()
            plt.pause(0.1)
            plt.clf()
