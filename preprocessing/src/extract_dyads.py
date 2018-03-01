import sys
import argparse

import matplotlib.pyplot as plt
from pathlib import Path
import datetime

import cv2
import numpy as np

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def find_video_reference_time(vid, folder):
    """
    Find the ellapsed time corresponding the the beggining of the given video.
        input: 
            vid: the id of the video
            folder: the folder (date) containing the video
        output: 
            total_time: the total time in second corresponding to the beginning of this video
            video_duration: the duration of this video
    """
    duration_file = folder + '/durations.dat'
    begining, video_duration = 0, 0
    with open(duration_file, 'r') as f:
        for line in f.readlines():
            line_vid, duration = line.split()
            if line_vid == vid:
                video_duration = float(duration)
                break
            begining += float(duration)
    return begining, video_duration


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


def find_acceptable_portion(trajectory):
    """
    Find the portion of the trajectory that correspond to an interpretable portion of the video
        input: all the trajectory points
        output: the trajectory points inside the interpretable portion
    """
    min_x, max_x = 40000, 45000
    min_y, max_y = 3000, 6000
    acceptable_portion = []
    acceptable = False
    n = len(trajectory)
    for i in range(n):
        x, y = trajectory[i][1], trajectory[i][2]
        if not acceptable:
            if x > min_x and x < max_x and y > min_y and y < max_y:
                acceptable = True
                acceptable_portion.append(trajectory[i])
        else:
            if not (x > min_x and x < max_x and y > min_y and y < max_y):
                break
            else:
                acceptable_portion.append(trajectory[i])
    return acceptable_portion


def main(argv):
    parser = argparse.ArgumentParser(description='Extract dyads video from a video.')
    parser.add_argument('video', metavar='video_path', type=str,
                        help='the video to split')

    args = parser.parse_args()
    
    # definition of reference time
    folder = args.video.split('/')[-3]
    video_id = args.video.split('/')[-2]

    beginning, video_duration = find_video_reference_time(video_id, "videos/" + folder)
    video_beginning = datetime.datetime(2010, 10, 6, 0, 0, 0) + datetime.timedelta(seconds=beginning)
    video_ending = video_beginning + datetime.timedelta(seconds=video_duration)

    video_time_ref = datetime.datetime(2010, 10, 6, 0, 0, 6) + datetime.timedelta(milliseconds=500)
    absolute_time_ref = datetime.datetime(2010, 10, 6, 10, 40, 48)
    delta_time = video_time_ref - absolute_time_ref

    
    # parse annotation to find all dyads
    dyads = [] # list of couple of ids for each dyad

    with open("./annotations/" + folder + "/ground_truth.csv") as f:
            for line in f:
                data = list(map(float, line.split(";")))
                if data[1] == 2 and (int(data[2]), int(data[0])) not in dyads:
                    # if 2 people in the group and the dyad not already in the list
                    dyads.append((int(data[0]), int(data[2])))


    # trajectory for one individual in each dyad
    dyads_trajectories = {}

    # extraction of one trajectory for each dyad
    for dyad in dyads:
        file_trajectory = "./trajectories/" + folder + "/crowd/path_" + str(dyad[0]) + ".csv"
        time, x, y = parse_trajectory(file_trajectory)
        video_time = []
        t_0 = (time[0] + delta_time)
        if t_0 < video_ending and t_0 > video_beginning: # only keeping trajectory from the input video
            for t in time:
                video_time.append((t + delta_time))
            dyads_trajectories[dyad[0]] = list(map(list, list(zip(video_time, x, y))))
            # plotting the trajectories on a 2D plan
            # plt.plot([c[0] for c in coords], [c[1] for c in coords])
            # plt.show()

    # calibration of the camera
    calibration_world_coordinates = np.array([
        [12121, 6023],    # A
        [8518, -13],      # B
        [19302, 6749],    # C
        [15679, -47],     # D
        [25828, 6684],    # E
        [25637, -35],     # F
        [35664, 6572],   # G
        [34864, -88],    # H
        [39402, 6520],   # I
        [39447, -75]    # J
        # [44470, 6511]    # K
    ], np.float32)

    # calibration_image_coordinates = np.array([
    #     [1302, 467],     # A
    #     [986, 462],      # B
    #     [1363, 488],     # C
    #     [921, 480],      # D
    #     [1379, 515],     # E
    #     [774, 518],      # F
    #     [1440, 603],     # G
    #     [474, 598],      # H
    #     [1488, 684],     # I
    #     [147, 685],      # J
    #     [1669, 992]      # K
    # ], np.float32)

    calibration_image_coordinates = np.array([
        [1302, 516],     # A
        [986, 508],      # B
        [1366, 550],     # C
        [920, 534],      # D
        [1382, 598],     # E
        [776, 594],      # F
        [1438, 776],     # G
        [474, 738],      # H
        [1498, 950],     # I
        [150, 894]       # J
    ], np.float32)

    homography_matrix, mask = cv2.findHomography(
        calibration_world_coordinates, 
        calibration_image_coordinates
    )

    # split the video into corresponding sub-videos
    subvideo_times = {} # contains ref for start time and stop time
    for pedestrian_id, trajectory in dyads_trajectories.items():
        trajectory = find_acceptable_portion(trajectory)
        if len(trajectory) != 0:
            t_init = (trajectory[0][0] - video_beginning).total_seconds()
            if t_init < video_duration:
                t_final = (trajectory[-1][0] - video_beginning).total_seconds()
                subvideo_path = 'videos/' + folder + '/' + video_id + '/dyads/' + str(pedestrian_id) + '.avi'
                subvideo_times[pedestrian_id] = (t_init, t_final)
                for j in range(len(trajectory)):
                    # change time relatively to the sub_video
                    trajectory[j][0] = (trajectory[j][0] - video_beginning).total_seconds() - t_init
                dyads_trajectories[pedestrian_id] = trajectory
                ffmpeg_extract_subclip(args.video, t_init, t_final, subvideo_path)

    # # compute image trajectory using the homography matrix
    # j = 10533200
    # time_lists = []
    # image_trajectories = {}
    # for pedestrian_id, traj in dyads_trajectories.items():
    #     world_traj = []
    #     time_list = []
    #     for t, x, y in traj:
    #         world_traj.append([x, y])
    #         time_list.append(t)
        
    #     # plot world trajectory and coordinates of the sensors
    #     # plt.plot([tr[0] for tr in world_traj], [tr[1] for tr in world_traj])
    #     # plt.plot([c[0] for c in calibration_world_coordinates], [c[1] for c in calibration_world_coordinates], 'o')
    #     # plt.show()

    #     # image_traj, _ = cv2.projectPoints(
    #     #     np.array(world_traj), rvecs[0], tvecs[0], 
    #     #     camera_matrix, dist_coeffs
    #     # )
    #     world_traj = np.array(world_traj, dtype=np.float32)
    #     image_traj = cv2.perspectiveTransform(world_traj[None, :, :], homography_matrix)
    #     image_traj = image_traj.tolist()[0]
    #     image_trajectories[pedestrian_id] = [[time_list[i]] + image_traj[i] for i in range(len(image_traj))]


    # traj = image_trajectories[j]
    # world_traj = dyads_trajectories[j]
    # # for t, _, _ in traj:
    # #     print(t)

    # # compute projection of the sensor references
    # sensor_points = cv2.perspectiveTransform(calibration_world_coordinates[None, :, :], homography_matrix)
    # sensor_points = sensor_points.tolist()[0]

    # # plt.ion()
    # # plt.scatter([c[0] for c in calibration_world_coordinates], [c[1] for c in calibration_world_coordinates])

    # # read the video
    # frame_id = 0
    # coord_id = 0

    # dyad_cap = cv2.VideoCapture('./videos/00000_dyads/' + str(j) + '.avi')
    # fps = 29.97

    # while(True):
    #     # capture frame-by-frame
    #     ret, frame = dyad_cap.read()
        
    #     frame_time = frame_id / fps

    #     if traj[coord_id + 1][0] < frame_time:
    #         coord_id += 1

    #     # print(frame_time, traj[coord_id][0])

    #     # adding trajectory point
    #     cv2.circle(frame, (int(traj[coord_id][1]), int(traj[coord_id][2])), 10, (0,255,0), -1)
    #     # plt.scatter(int(world_traj[coord_id][1]), int(world_traj[coord_id][2]), 2, 'g')
    #     # plt.pause(0.001)

    #     # adding sensor reference
    #     for point in sensor_points:
    #         cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0,0,255), -1) 
        
    #     resized_frame = cv2.resize(frame, dsize=(0, 0), fx=1/2, fy=1/2)

    #     # display the resulting frame
    #     cv2.imshow('frame', resized_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     frame_id += 1
    # # When everything done, release the capture
    # dyad_cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[:1]) # first arg is script name
        