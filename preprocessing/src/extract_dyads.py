import matplotlib.pyplot as plt
from pathlib import Path
import datetime

import cv2
import numpy as np

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
        # plotting the trajectories on a 2D plan
        # plt.plot([c[0] for c in coords], [c[1] for c in coords])
        # plt.show()

# computation of the homography and plot of the trajectories on the video
sensor_world_coordinates = np.array([
    [12121, 6023, 0],   # A
    [8518, -13, 0],     # B
    [19302, 6749, 0],   # C
    [15679, -47, 0],    # D
    [25828, 6684, 0],   # E
    [25637, -35, 0],    # F
    [35664, 6572, 0],   # G
    [34864, -88, 0],    # H
    [39402, 6520, 0],   # I
    [39447, -75, 0],    # J
    [44470, 6511, 0]    # K
], np.float32)

# sensor_image_coordinates = np.array([
#     [1302, 467],     # A
#     [986, 462],     # B
#     [1363, 488],     # C
#     [921, 480],      # D
#     [1379, 515],     # E
#     [774, 518],      # F
#     [1440, 603],     # G
#     [474, 598],      # H
#     [1488, 684],     # I
#     [147, 685],      # J
#     [1669, 992]      # K
#  ], np.float32)

sensor_image_coordinates = np.array([
    [1359, 382],     # A
    [1039, 381],     # B
    [1420, 403],     # C
    [974, 399],      # D
    [1437, 429],     # E
    [828, 439],      # F
    [1499, 462],     # G
    [528, 466],      # H
    [1546, 524],     # I
    [205, 540],      # J
    [1731, 736]      # K
], np.float32)

# reading the video
vcap = cv2.VideoCapture('./videos/00000.AVI')

if vcap.isOpened(): 
    # get vcap property 
    width = int(vcap.get(3))
    height = int(vcap.get(4))

retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    [sensor_world_coordinates], [sensor_image_coordinates],
    (width, height), None, None
)

# splitting of the video into corresponding sub-videos
i = 0
video_times = {} # contains ref for start time and stop time
for trajectory in dyads_trajectories:
    t_init = (trajectory[0][0] - video_beginning).total_seconds()
    t_final = (trajectory[0][len(trajectory[0])-1] - video_beginning).total_seconds()
    duration = t_final - t_init
    subvideo_name = "00000_dyads/sub_" + str(i) + ".avi"
    video_times[i] = (t_init, t_final)
    # ffmpeg_extract_subclip("./videos/00000.AVI", t_init, t_final, "./videos/" + subvideo_name)
    i += 1

traj = dyads_trajectories[0]

image_points, jacobian = cv2.projectPoints(
    sensor_world_coordinates, rvecs[0], tvecs[0], 
    camera_matrix, dist_coeffs
)

dyad_cap = cv2.VideoCapture('./videos/00000_dyads/sub_0.avi')

# read the video
while(True):
    # capture frame-by-frame
    ret, frame = dyad_cap.read()

    # adding points to the frame
    for point in sensor_image_coordinates:
        cv2.circle(frame, (point[0], point[1]), 10, (0,255,0)) 
        resized_frame = cv2.resize(frame, dsize=(0, 0), fx=1/2, fy=1/2)

    # display the resulting frame
    cv2.imshow('frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
dyad_cap.release()
cv2.destroyAllWindows()

    