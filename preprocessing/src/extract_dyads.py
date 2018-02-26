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
    time, x, y = [], [], []
    with open(file_path) as f:
        n = int(f.readline())
        for _ in range(n):
            line = f.readline().split(";")
            ms_time = int(float(line[0]))
            s_time = int(float(line[0])/1000)
            ms = ms_time - s_time * 1000
            time.append(datetime.datetime.fromtimestamp(
                s_time) + datetime.timedelta(milliseconds=ms)
            )
            x.append(float(line[1]))
            y.append(float(line[2]))
    return time, x, y


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
    time, x, y = parse_trajectory(file_trajectory)
    video_time = []
    t_0 = (time[0] + delta_time)
    if t_0 < video_ending: # for the first video
        for t in time:
            video_time.append((t + delta_time))
        dyads_trajectories.append(list(zip(video_time, x, y)))
        # plotting the trajectories on a 2D plan
        # plt.plot([c[0] for c in coords], [c[1] for c in coords])
        # plt.show()

# calibration of the camera
calibration_world_coordinates = np.array([
    [12121, 6023, 0],    # A
    [8518, -13, 0],      # B
    [19302, 6749, 0],    # C
    [15679, -47, 0],     # D
    [25828, 6684, 0],    # E
    [25637, -35, 0],     # F
    [35664, 6572, 0],   # G
    [34864, -88, 0],    # H
    [39402, 6520, 0],   # I
    [39447, -75, 0],    # J
    [44470, 6511, 0]    # K
], np.float32)

calibration_image_coordinates = np.array([
    [1302, 467],     # A
    [986, 462],      # B
    [1363, 488],     # C
    [921, 480],      # D
    [1379, 515],     # E
    [774, 518],      # F
    [1440, 603],     # G
    [474, 598],      # H
    [1488, 684],     # I
    [147, 685],      # J
    [1669, 992]      # K
], np.float32)

# reading the video
vcap = cv2.VideoCapture('./videos/00000.AVI')

if vcap.isOpened(): 
    # get vcap property 
    width = int(vcap.get(3))
    height = int(vcap.get(4))

retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    [calibration_world_coordinates], [calibration_image_coordinates],
    (width, height), None, None
)

# splitting of the video into corresponding sub-videos
i = 0
subvideo_times = {} # contains ref for start time and stop time
for trajectory in dyads_trajectories:
    t_init = (trajectory[0][0] - video_beginning).total_seconds()
    t_final = (trajectory[-1][0] - video_beginning).total_seconds()
    duration = t_final - t_init
    subvideo_name = "00000_dyads/sub_" + str(i) + ".avi"
    subvideo_times[i] = (t_init, t_final)
    for j in range(len(trajectory)):
        # change time relatively to the sub_video
        trajectory[j] += ((trajectory[j][0] - video_beginning).total_seconds() - t_init,)
    # ffmpeg_extract_subclip("./videos/00000.AVI", t_init, t_final, "./videos/" + subvideo_name)
    i += 1

# plotting trajectories for the video j (dyad j)
j = 2
fps = 29.97
time_lists = []
image_trajectories = []
for traj in dyads_trajectories:
    world_traj = []
    time_list = []
    for _, x, y, t in traj:
        world_traj.append([x, y, 0.0])
        time_list.append(t)
    
    # plot world trajectory and coordinates of the sensors
    # plt.plot([tr[0] for tr in world_traj], [tr[1] for tr in world_traj])
    # plt.plot([c[0] for c in calibration_world_coordinates], [c[1] for c in calibration_world_coordinates], 'o')
    # plt.show()

    image_traj, _ = cv2.projectPoints(
        np.array(world_traj), rvecs[0], tvecs[0], 
        camera_matrix, dist_coeffs
    )
    image_traj = [i[0].tolist() for i in image_traj]
    image_trajectories.append([[time_list[i]] + image_traj[i] for i in range(len(image_traj))])


dyad_cap = cv2.VideoCapture('./videos/00000_dyads/sub_' + str(j) + '.avi')
traj = image_trajectories[j]

# compute projection of the sensor references
sensor_points, _ = cv2.projectPoints(
        calibration_world_coordinates, rvecs[0], tvecs[0], 
        camera_matrix, dist_coeffs
)
sensor_points = [i[0].tolist() for i in sensor_points]

# read the video
frame_id = 0
coord_id = 0
while(True):
    # capture frame-by-frame
    ret, frame = dyad_cap.read()
    
    frame_time = frame_id / fps

    if traj[coord_id + 1][0] < frame_time:
        coord_id += 1
    # adding trajectory point
    cv2.circle(frame, (int(traj[coord_id][1]), int(traj[coord_id][2])), 10, (0,255,0), -1) 

    # adding sensor reference
    for point in sensor_points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, (0,0,255), -1) 
    
    resized_frame = cv2.resize(frame, dsize=(0, 0), fx=1/2, fy=1/2)

    # display the resulting frame
    cv2.imshow('frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_id += 1
# When everything done, release the capture
dyad_cap.release()
cv2.destroyAllWindows()

    