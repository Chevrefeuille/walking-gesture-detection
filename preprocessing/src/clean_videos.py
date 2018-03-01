import os
from pathlib import Path
from shutil import copyfile

if __name__ == "__main__":
    p = Path('videos/dyads/').glob('**/*.avi')
    for path in p: # each video folder
        video_path = str(path)
        print('Removing ' + video_path)
        os.remove(video_path)

    folder = '2010_10_06'
    p = Path('videos/' + folder)
    for path in p.iterdir(): # each video folder
        if path.is_dir():
            dyads_folder = str(path) 
            dp = Path(dyads_folder)
            for v_path in dp.glob('dyads/*.avi'): # each subvideo
                video_path = str(v_path)
                print('Removing ' + video_path)
                os.remove(video_path)
                        

