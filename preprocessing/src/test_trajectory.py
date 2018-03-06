import sys
import argparse
import subprocess as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

import cv2
import numpy as np

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def fetch_time_ref(vid, folder):
    """
    Fetch the time references for the video.
        input: 
            vid: the id of the video
            folder: the folder (date) containing the video
        output: 
            v: a video datetime
            w: a wordl datetime
    """
    ref_file = folder + '/time_references.dat'
    begining, video_duration = 0, 0
    v, w = None, None
    with open(ref_file, 'r') as f:
        for line in f.readlines():
            line_vid, video_time, ms, world_time = line.split()
            if line_vid == vid:
                t1 = list(map(int, video_time.split(",")))
                t2 = list(map(int, world_time.split(",")))
                v = datetime.datetime(t1[0], t1[1], t1[2], t1[3], t1[4], t1[5])
                v += datetime.timedelta(milliseconds=int(ms))
                w = datetime.datetime(t2[0], t2[1], t2[2], t2[3], t2[4], t2[5])
    return v, w

def find_video_reference_time(vid, folder):
    """
    Find the elapsed time corresponding the the beggining of the given video.
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


def parse_trajectory(file_path, delta_time, video_beginning):
    """
    Parse the trajectory contained in the given file
        input: file path as a string
        output: lists containing the time and (x, y) couples
    """
    t, x, y = [], [], []
    with open(file_path) as f:
        n = int(f.readline())
        for _ in range(n):
            line = f.readline().split(";")
            ms_time = int(float(line[0]))
            s_time = int(float(line[0])/1000)
            ms = ms_time - s_time * 1000
            absolute_time = datetime.datetime.fromtimestamp(s_time) + datetime.timedelta(milliseconds=ms)
            video_time = absolute_time + delta_time
            video_time_seconds = (video_time - video_beginning).total_seconds()
            x.append(float(line[1]))
            y.append(float(line[2]))
            t.append(video_time_seconds)
    return t, x, y

def compute_middle_trajectory(t_1, x_1, y_1, t_2, x_2, y_2):
    """
    Compute the trajetory of the middle between the two trajectory when
    the times match
    input: 
        t_1, x_1, y_1: time and coordinates lists for the first trajectory
        t_2, x_2, y_2: time and coordinates lists for the second trajectory
    output: the list of triplets (t, x, y) for the middle point
    """
    t_0 = max(t_1[0], t_2[0])
    t_end = min(t_1[-1], t_2[-1])
    i_0_1, i_end_1 = 0, 0
    for i in range(len(t_1)):
        if t_1[i] <= t_0:
            i_0_1 = i
        elif t_1[i] >= t_end:
            i_end_1 = i
            break
    i_0_2, i_end_2 = 0, 0
    for i in range(len(t_2)):
        if t_2[i] <= t_0:
            i_0_2 = i
        elif t_2[i] >= t_end:
            i_end_2 = i
            break
    t_middle = t_1[i_0_1:i_end_1]
    x_1, y_1 = x_1[i_0_1:i_end_1], y_1[i_0_1:i_end_1]
    x_2, y_2 = x_2[i_0_2:i_end_2], y_2[i_0_2:i_end_2]
    x_middle = [(x_1[i] + x_2[i]) / 2 for i in range(min(len(x_1), len(x_2)))]
    y_middle = [(y_1[i] + y_2[i]) / 2 for i in range(min(len(y_1), len(y_2)))]
    middle_traj = [[t_middle[i], x_middle[i], y_middle[i]] for i in range(min(len(x_1), len(x_2)))]
    return middle_traj

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

def compute_z_coordinate(x, y):
    """
    Compute the z coordinate according to the estimated map of the corridor
        input: x, y coordinates
        output: z coordinates
    """
    slope = 400 / (35500 - 26000)
    b = - slope * 26000
    if x < 26000:
        return 0
    elif x > 35500:
        return 400
    else:
        return slope * x + b

def main(argv):
    height, width = 1080, 1920
    parser = argparse.ArgumentParser(description='Extract dyads video from a video.')
    parser.add_argument('video', metavar='video_path', type=str,
                        help='the video to split')

    args = parser.parse_args()
    
    folder = args.video.split('/')[-3]
    video_id = args.video.split('/')[-2]

    beginning, video_duration = find_video_reference_time(video_id, "videos/" + folder)
    ending = beginning + video_duration
    video_beginning = datetime.datetime(2010, 10, 6, 0, 0, 0) + datetime.timedelta(seconds=beginning)

    # definition of reference times
    video_time_ref, absolute_time_ref = fetch_time_ref(video_id, "videos/" + folder)

    delta_time = video_time_ref - absolute_time_ref
    
    ids = [] 

    with open("./annotations/" + folder + "/ground_truth.csv") as f:
        for line in f:
            data = list(map(float, line.split(";")))
            ids.append(int(data[0]))


    # trajectories
    trajectories = {}

    # extraction of one trajectory for each dyad
    for pid in ids:
        file_traj = "./trajectories/" + folder + "/crowd/path_" + str(pid) + ".csv"
        t, x, y = parse_trajectory(file_traj, delta_time, video_beginning)
        if t[0] > 0 and t[0] < ending: # only keeping trajectory from the input video
            trajectories[pid] = [[t[i], x[i], y[i]] for i in range(len(x))]
            # plotting the trajectories on a 2D plan
            # plt.plot(x_1, y_1)
            # plt.plot(x_2, y_2)
            # plt.plot([c[1] for c in middle_traj], [c[2] for c in middle_traj])
            # plt.show()
            # break

    # calibration of the camera
    calibration_world_coordinates = np.array([
        [12121, 6023, 0],    # A
        [8518, -13, 0],      # B
        [19302, 6749, 0],    # C
        [15679, -47, 0],     # D
        [25828, 6684, 0],    # E
        [25637, -35, 0],     # F
        [35664, 6572, 400],   # G
        [34864, -88, 400],    # H
        [39402, 6520, 400],   # I
        [39447, -75, 400]    # J
        # [44470, 6511]    # K
    ], np.float32)

    calibration_test_coordinates = np.array([
        [12121, 6023, 850],    # A
        [8518, -13, 850],      # B
        [19302, 6749, 850],    # C
        [15679, -47, 850],     # D
        [25828, 6684, 850],    # E
        [25637, -35, 850],     # F
        [35664, 6572, 1250],   # G
        [34864, -88, 1250],    # H
        [39402, 6520, 1250],   # I
        [39447, -75, 1250]    # J
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
        [1438, 690],     # G
        [474, 662],      # H
        [1488, 798],     # I
        [146, 784]       # J
    ], np.float32)

    camera_matrix = np.array([
        [2200.0, 0, width / 2.0],
        [0, 2200.0, height / 2.0],
        [0, 0, 1]
    ])

    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [calibration_world_coordinates], [calibration_image_coordinates],
        (width, height), camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )


    valid_trajectories = {} # trajectories consistent with the input video

    for pedestrian_id, trajectory in trajectories.items():
        if len(trajectory) != 0:
            t_init = trajectory[0][0]
            if t_init < video_duration:
                valid_trajectories[pedestrian_id] = trajectory
                
    
    # compute image trajectory and boundind boxes using the homography matrix
    image_trajectories = {}
    image_trajectories2 = {}
    bbs = {}
    for pedestrian_id, traj in valid_trajectories.items():
        world_traj, world_bl_bb, world_tr_bb = [], [], []
        world_traj2 = []
        time_list = []
        for t, x, y in traj:
            world_traj.append([x, y, compute_z_coordinate(x, y)])
            world_traj2.append([x, y, 0])
            world_bl_bb.append([x, y - 400, compute_z_coordinate(x, y)])
            world_tr_bb.append([x, y + 400, compute_z_coordinate(x, y) + 2000])
            time_list.append(t)

        # plot world trajectory and coordinates of the sensors
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter([tr[0] for tr in world_traj], [tr[1] for tr in world_traj], [tr[2] for tr in world_traj])
        # # plt.plot([c[0] for c in calibration_world_coordinates], [c[1] for c in calibration_world_coordinates], 'o')
        # plt.show()

        # image_traj, _ = cv2.projectPoints(
        #     np.array(world_traj), rvecs[0], tvecs[0], 
        #     camera_matrix, dist_coeffs
        # )

        # middle point
        world_traj = np.array(world_traj, dtype=np.float32)
        image_traj = cv2.projectPoints(
            world_traj,
            rvecs[0], tvecs[0], camera_matrix, dist_coeffs)[0]
        image_traj = [i[0].tolist() for i in image_traj]

        world_traj2 = np.array(world_traj2, dtype=np.float32)
        image_traj2 = cv2.projectPoints(
            world_traj2,
            rvecs[0], tvecs[0], camera_matrix, dist_coeffs)[0]
        image_traj2 = [i[0].tolist() for i in image_traj2]

        # bottom left corner of the bounding box
        world_bl_bb = np.array(world_bl_bb, dtype=np.float32)
        image_bl_bb = cv2.projectPoints(
            world_bl_bb,
            rvecs[0], tvecs[0], camera_matrix, dist_coeffs)[0]
        image_bl_bb = [i[0].tolist() for i in image_bl_bb]

        # top right corner of the bounding box
        world_tr_bb = np.array(world_tr_bb, dtype=np.float32)
        image_tr_bb = cv2.projectPoints(
            world_tr_bb,
            rvecs[0], tvecs[0], camera_matrix, dist_coeffs)[0]
        image_tr_bb = [i[0].tolist() for i in image_tr_bb]
        
        image_trajectories[pedestrian_id] = [[time_list[i]] + image_traj[i] for i in range(len(image_traj))]
        image_trajectories2[pedestrian_id] = [[time_list[i]] + image_traj2[i] for i in range(len(image_traj2))]
        bbs[pedestrian_id] = [image_bl_bb, image_tr_bb]

    for pedestrian_id in image_trajectories:
        traj = image_trajectories[pedestrian_id]
        traj2 = image_trajectories2[pedestrian_id]
        world_traj = trajectories[pedestrian_id]
        bb = bbs[pedestrian_id]
    
        # compute projection of the sensor references
        # sensor_points, jacobian = cv2.projectPoints(
        #     calibration_test_coordinates,
        #     rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
        # sensor_points = [s[0].tolist() for s in sensor_points]
        # # plt.ion()
        # # plt.scatter([c[0] for c in calibration_world_coordinates], [c[1] for c in calibration_world_coordinates])

        frame_id = 0
        coord_id = 0

        cap = cv2.VideoCapture(args.video)

        t_0 = traj[0][0]
        cap.set(cv2.CAP_PROP_POS_MSEC, t_0 * 1000) 

        fps = cap.get(5)

        while(True):
            # capture frame-by-frame
            ret, frame = cap.read()
            if not type(frame) is np.ndarray:
                break
            frame_time = frame_id / fps + t_0
            if coord_id + 1 >= len(traj):
                break
            if traj[coord_id + 1][0] < frame_time:
                coord_id += 1

            # print(frame_time, traj[coord_id][0])

            # adding trajectory point
            # for coord in traj:
            #     circle_i = min(width, max(0, int(coord[1])))
            #     circle_j = min(height, max(0, int(coord[2])))
            #     cv2.circle(frame, (circle_i, circle_j), 1, (0,255,0), -1)
    

            # for i in range(len(bb[0])):
            x1 = min(width, max(0, int(bb[0][coord_id][0])))
            y1 = min(width, max(0, int(bb[0][coord_id][1])))
            x2 = min(width, max(0, int(bb[1][coord_id][0])))
            y2 = min(width, max(0, int(bb[1][coord_id][1])))
            cv2.rectangle(frame, 
                (x1, y1), (x2, y2),
                255, thickness=1)

            circle_i = min(width, max(0, int(traj[coord_id][1])))
            circle_j = min(height, max(0, int(traj[coord_id][2])))
            cv2.circle(frame, (circle_i, circle_j), 5, (0,0,255), -1)

            

            # plt.scatter(int(world_traj[coord_id][1]), int(world_traj[coord_id][2]), 2, 'g')
            # plt.pause(0.001)

            # adding sensor reference
            # for point in sensor_points:
                # cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0,0,255), -1) 
                        
            resized_frame = cv2.resize(frame, dsize=(0, 0), fx=1/2, fy=1/2)

            # display the resulting frame
            cv2.imshow('frame', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1
        # When everything done, release the capture
        cap.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[:1]) # first arg is script name
        