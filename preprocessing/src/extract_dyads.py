import matplotlib.pyplot as plt
from pathlib import Path
import datetime

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def parse_trajectory(file_path):
    """
    Parse the trajectory contained in the given file
        input: file path as a string
        output: lists containing the time and (x, y) couples
    """
    time, coords = [], []
    with open(file_path) as f:
        n = int(f.readline())
        for _ in range(n):
            line = f.readline().split(";")
            time.append(datetime.datetime.fromtimestamp(
                int(float(line[0])/1000)
            ))
            coords.append((float(line[1]), float(line[2])))
    return time, coords


folder = "2010_10_06"
video_time_ref = datetime.datetime(2010, 10, 6, 0, 0, 17)
absolute_time_ref = datetime.datetime(2010, 10, 6, 10, 40, 55)
delta_time = video_time_ref - absolute_time_ref

video_beginning = datetime.datetime(2010, 10, 6, 0, 0, 0)
video_ending = datetime.datetime(2010, 10, 6, 0, 16, 45)


dyads = [] # list of couple of ids for each dyad

with open("./annotations/" + folder + "/ground_truth.csv") as f:
        for line in f:
            data = list(map(float, line.split(";")))
            if data[1] == 2 and (int(data[2]), int(data[0])) not in dyads:
                # if 2 people in the group and the dyad not already in the list
                dyads.append((int(data[0]), int(data[2])))


dyads_trajectories = []

# extraction of the correspondant trajectories for the 2 people
for dyad in dyads:
    file_trajectory = "./trajectories/" + folder + "/crowd/path_" + str(dyad[0]) + ".csv"
    time, coords = parse_trajectory(file_trajectory)
    video_time = []
    t_0 = (time[0] + delta_time)
    if t_0 < video_ending: # for the first video
        for t in time:
            video_time.append((t + delta_time))
        dyads_trajectories.append([video_time, coords])
        plt.plot([c[0] for c in coords], [c[1] for c in coords])
        plt.show()

# splitting of the video into corresponding sub-videos
i = 0
for trajectory in dyads_trajectories:
    t_init = (trajectory[0][0] - video_beginning).total_seconds()
    t_final = (trajectory[0][len(trajectory[0])-1] - video_beginning).total_seconds()
    duration = t_final - t_init
    ffmpeg_extract_subclip("./videos/00000.AVI", t_init, t_final, "./videos/00000_dyads/sub_"+str(i)+".avi")
    i += 1
    