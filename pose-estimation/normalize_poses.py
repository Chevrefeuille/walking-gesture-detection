import os
from pathlib import Path
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


if __name__ == "__main__":
    # folder that contains the st-gcn files (one per video)
    stgcn_json_path =  '../data/poses/normalized_st-gcn_format/'

    p = Path(stgcn_json_path)
    
    for path in p.glob('*.json'):
        json_path = str(path)
        if 'consistent' in json_path:
            data = json.load(open(json_path))
            frames, normalized_frames = data['data'],  []
            ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
            skeletons_data = []
            n_frames = len(frames)
            for frame in ordered_frames:
                normalized_frame = frame
                skeletons, normalized_skeletons = frame['skeleton'], []
                list_x, list_y = [], []
                min_x, min_y = 1, 1
                max_x, max_y = 0, 0
                n_skeletons = len(skeletons)
                for skeleton in skeletons:
                    coordinates = skeleton['pose']
                    for i in range(0, len(coordinates), 2):
                        x, y = coordinates[i], 1 - coordinates[i+1]
                        if x != 0 and x < min_x:
                            min_x = x
                        if x != 0 and x > max_x:
                            max_x = x
                        if y != 1 and y < min_y:
                            min_y = y
                        if y != 1 and y > max_y:
                            max_y = y

                for skeleton in skeletons:
                    normalized_skeleton = skeleton
                    coordinates = skeleton['pose']
                    normalized_coordinates = []
                    for i in range(0, len(coordinates), 2):
                        x, y = coordinates[i], 1 - coordinates[i+1]
                        if x != 0:
                            x = (x - min_x) / (max_x - min_x)
                        if y !=  0:
                            y = (y - min_y) / (max_y - min_y)
                        normalized_coordinates += [x, y]
                    normalized_skeleton['pose'] = normalized_coordinates
                    normalized_skeletons += [normalized_skeleton]
                    normalized_frame['skeleton'] = normalized_skeletons

                normalized_frames += [normalized_frame]
            data['data'] = normalized_frames

            with open(json_path, 'w') as outfile:
                print('Normalizing {}'.format(json_path))
                json.dump(data, outfile)
                
                        
                    
            