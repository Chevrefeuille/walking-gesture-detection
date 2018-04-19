import os
from pathlib import Path
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse

def update_plot(i, data, scat):
    scat.set_offsets(data[i])
    return scat,

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pose checker.')
    # folder that will contain the converted files (one per video)
    parser.add_argument('--json_path', default='../ data/poses/normalized_st-gcn_format/10533200.json')

    arg = parser.parse_args()

    data = json.load(open(arg.json_path))
    frames = data['data']
    ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
    skeletons_data = []
    n_frames = len(frames)
    for frame in ordered_frames:
        # print(frame['frame_index'])
        frame_skeletons = []
        skeletons = frame['skeleton']
        for skeleton in skeletons:
            coordinates = skeleton['pose']
            for i in range(0, len(coordinates), 2):
                frame_skeletons += [[coordinates[i], coordinates[i+1]]]
        skeletons_data.append(frame_skeletons)
    
    # display the skeletons animation
    fig = plt.figure()
    scat = plt.scatter(skeletons_data[0][0], skeletons_data[0][1])
    ani = animation.FuncAnimation(fig, update_plot, frames=range(n_frames),
                                    fargs=(skeletons_data, scat),interval=10, repeat=False)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()
