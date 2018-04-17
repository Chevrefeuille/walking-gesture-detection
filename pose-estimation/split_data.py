import os
from pathlib import Path


if __name__ == "__main__":
    # folder that contains the st-gcn files (one per video)
    stgcn_json_path =  '../data/poses/normalized_st-gcn_format/'

    train_ratio = 0.6

    p = Path(stgcn_json_path)
    
    for path in p.glob('*.json'):
        json_path = str(path)

            