import os
from pathlib import Path

if __name__ == "__main__":
    p = Path('../data/poses/openpose_format')
    for path in p.glob('*.json'): # each json file
        json_path = str(path)
        print('Removing ' + json_path)
        os.remove(json_path)
    
    p = Path('../data/poses/st-gcn_format')
    for path in p.glob('*.json'): # each json file
        json_path = str(path)
        print('Removing ' + json_path)
        os.remove(json_path)
                        

