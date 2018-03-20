import os
from pathlib import Path
import json
from pprint import pprint

if __name__ == "__main__":
    folder = '2010_10_06'
    p = Path('../data/poses/openpose_format')
    list_data_file = '../data/data_list.dat'
    labels = {}
    with open(list_data_file, 'r') as f:
        for line in f:
            vid  = int(line)
            j = 0               
            stgcn_data_array = []
            stgcn_data = {}
            dest_path = '../data/poses/st-gcn_format/' + str(vid) + '.json'
            for path in p.glob(str(vid) + '*.json'): # each json file
                json_path = str(path)
                j += 1
                frame_id = int(((json_path.split('/')[-1]).split('.')[0]).split('_')[1])
                frame_data = {'frame_index': frame_id}
                data = json.load(open(json_path))
                skeletons = []        
                for person in data['people']:
                    score, coordinates = [], []
                    skeleton = {}
                    keypoints = person['pose_keypoints_2d']
                    for i in range(0,len(keypoints),3):
                        coordinates +=  [keypoints[i], keypoints[i + 1]]
                        score += [keypoints[i + 2]]
                    skeleton['pose'] = coordinates
                    skeleton['score'] = score
                    skeletons += [skeleton]
                frame_data['skeleton'] = skeletons
                stgcn_data_array += [frame_data]
            if j == 300:
                labels[str(vid)] = {"has_skeleton": True, 
                    "label": "fake_label", 
                    "label_index": 0}
                stgcn_data['data'] = stgcn_data_array
                stgcn_data['label'] = 'fake_label'
                stgcn_data['label_index'] = 0
                with open(dest_path, 'w') as outfile:
                    json.dump(stgcn_data, outfile)

    labels_file = '../data/poses/fake_labels.json'
    with open(labels_file, 'w') as label_file:
        json.dump(labels, label_file)