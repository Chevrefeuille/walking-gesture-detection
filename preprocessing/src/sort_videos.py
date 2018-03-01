import os
from pathlib import Path
from shutil import copyfile

def find_class(pedestrian_id, folder):
    """
    Using the ground truth, find the gestures performed by the pedestrian.
    """
    with open("./annotations/" + folder + "/ground_truth.csv") as f:
        for line in f:
            data = list(map(float, line.split(";")))
            if data[0] == pedestrian_id:
                return data[14:18]


if __name__ == "__main__":
    folder = '2010_10_06'
    gesture_map = {
        0: 'friend_gaze',
        1: 'speak',
        2: 'target_gaze',
        3: 'contact'
    }
    p = Path('videos/' + folder)
    for path in p.iterdir(): # each video folder
        if path.is_dir():
            dyads_folder = str(path) 
            dp = Path(dyads_folder)
            for v_path in dp.glob('dyads/*.avi'): # each subvideo
                video_path = str(v_path)
                pedestrian_id = int((video_path.split('/')[-1]).split('.')[0])
                gestures = find_class(pedestrian_id, folder)
                no_gesture = True
                for i in range(4):
                    if gestures[i]:
                        no_gesture = False
                        dest_path = 'videos/dyads/gestures/' + gesture_map[i] + '/' + str(pedestrian_id) + '.avi'
                        copyfile(video_path, dest_path)
                if no_gesture:
                    dest_path = 'videos/dyads/no_gestures/' + str(pedestrian_id) + '.avi'
                    copyfile(video_path, dest_path)
                        

