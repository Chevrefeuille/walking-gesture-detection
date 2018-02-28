from pathlib import Path
from moviepy.editor import VideoFileClip


if __name__ == "__main__":
    folder = 'videos/2010_10_06'
    durations = {}
    p = Path(folder)
    for path in p.iterdir():
        if path.is_dir():
            path_in_str = str(path)
            video_id = path_in_str.split('/')[-1]
            video_path = path_in_str + '/' + video_id + '.avi'
            clip = VideoFileClip(video_path)
            durations[video_id] = clip.duration

    duration_file = folder + '/durations.dat'
    with open(duration_file, 'w') as f:
        for vid, duration in durations.items():
            data = vid + '\t' + str(duration) + '\n'
            f.write(data)

