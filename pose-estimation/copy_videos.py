import os
from pathlib import Path
from shutil import copyfile

if __name__ == "__main__":
    folder = '2010_10_06'
    p = Path('../preprocessing/videos/' + folder)
    for path in p.iterdir(): # each video folder
        if path.is_dir():
            dyads_folder = str(path)
            for v_path in path.glob('dyads/*.avi'): # each subvideo
                video_path = str(v_path)
                pedestrian_id = int(((video_path.split('/')[-1]).split('.')[0]).split('_')[0])    
                dest_path = '../data/videos/' + str(pedestrian_id) + '.avi'
                print('Copying {} to {}'.format(video_path, dest_path))
                copyfile(video_path, dest_path)