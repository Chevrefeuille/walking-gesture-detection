from pathlib import Path

if __name__ == "__main__": 
    list_data_file = '../data/data_list.dat'
    with open(list_data_file, 'w') as f:
        p = Path('../data/videos/')
        for path in p.glob('*.avi'): # each videos
            video_path = str(path)
            pedestrian_id = int(((video_path.split('/')[-1]).split('.')[0]).split('_')[0])
            f.write(str(pedestrian_id) + '\n')