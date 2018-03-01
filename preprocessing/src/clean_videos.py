import os
from pathlib import Path
from shutil import copyfile

if __name__ == "__main__":
    p = Path('videos/dyads/').glob('**/*.avi')
    for path in p: # each video folder
        video_path = str(path)
        os.remove(video_path)
                        

