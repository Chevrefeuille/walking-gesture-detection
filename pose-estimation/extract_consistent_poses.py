import os
from pathlib import Path
import json
from pprint import pprint
import matplotlib.pyplot as plt
import itertools

def find_most_confident(skeletons):
    """
    Return the two skeletons with the largest average confidences and closest gravity center
    """
    data = []
    for skeleton in skeletons:
        n_joints = len(skeleton['score'])
        coordinates = skeleton['pose']
        average_confidence = sum(skeleton['score']) / n_joints
        x, y = [], []
        for i in range(0, len(coordinates), 2):
            x += [coordinates[i]]
            y += [coordinates[i+1]]
        center_of_gravity = (sum(x) / n_joints, sum(y) / n_joints)
        data += [(skeleton, average_confidence, center_of_gravity)]
    sorted_by_confidence = sorted(data, key=lambda c: c[1], reverse=True)
    best_s1, best_s2 = sorted_by_confidence[0][0], sorted_by_confidence[0][0]
    best_distance = 1
    for pair in itertools.combinations(sorted_by_confidence, 2):
        s1, _, g1 = pair[0]
        s2, _, g2 = pair[1]
        distance = ((g1[0] - g2[0])**2 + (g1[1] - g2[1])**2)**.5
        if distance < best_distance:
            best_s1, best_s2 = s1, s2
    
    return best_s1, best_s2

def find_consistent(skeletons, s1):
    """
    Return the skeleton which is consistent with the skeletons s1
    """
    average_distances = []
    for skeleton in skeletons:
        coordinates = skeleton['pose']
        average_distance = 0
        n_joints = len(skeleton['score'])
        for i in range(0, len(coordinates), 2):
            x1, y1 = s1['pose'][i], s1['pose'][i+1]
            x, y = coordinates[i], coordinates[i+1]
            d = ((x1 - x)**2 + (y1 - y)**2)**.5
            average_distance += d
        average_distance /= n_joints
        # print(average_distance)
        average_distances += [(skeleton, average_distance)]

    sorted_by_distance = sorted(average_distances, key=lambda c: c[1])
    return sorted_by_distance[0][0]


if __name__ == "__main__":
    # folder that contains the st-gcn files (one per video)
    stgcn_json_path =  '../data/poses/st-gcn_format/'
    consistent_json_path =  '../data/poses/normalized_st-gcn_format/'

    p = Path(stgcn_json_path)
    
    for path in p.glob('*.json'):
        json_path = str(path)
        if 'consistent' not in json_path:
            file_name = json_path.split('/')[-1]
            dest_path = '{}{}'.format(consistent_json_path, file_name)
            data = json.load(open(json_path))
            frames = data['data']
            ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
            first_frame = ordered_frames[0]
            first_skeletons = first_frame['skeleton']
            s1, s2 = find_most_confident(first_skeletons)
            consistent_frames = []
            for frame in ordered_frames:
                skeletons = frame['skeleton']
                if len(skeletons) >= 1:
                    consistent_1 = find_consistent(skeletons, s1)
                    skeletons.remove(consistent_1)
                if len(skeletons) >= 1:
                    consistent_2 = find_consistent(skeletons, s2)
                s1, s2 = consistent_1, consistent_2
                consistent_frames += [{
                    'skeleton': [consistent_1, consistent_2],
                    'frame_index': frame['frame_index']
                    }]
            data['data'] = consistent_frames
            with open(dest_path, 'w') as outfile:
                print('Creating {}'.format(dest_path))
                json.dump(data, outfile)

            
            
