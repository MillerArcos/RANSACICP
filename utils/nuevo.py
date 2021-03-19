#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:42:21 2021

@author: miller
"""
import os
import sys
import csv
import copy
import time
import random
import argparse
#import rospy
import random
import copy
import numpy as np
#from sensor_msgs.msg import PointCloud2, PointField
#from sensor_msgs import point_cloud2
import tila1 as convert
import numpy as np
np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

from minisam import *
from utils.ScanContextManager import *
from utils.PoseGraphManager import *
from utils.UtilsMisc import *
import utils.UtilsPointcloud as Ptutils
import utils.ICP as ICP


parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=5000) # 5000 is enough for real time

parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10) # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10) # same as the original paper

parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)

parser.add_argument('--data_base_dir', type=str, 
                    default='/home/miller/dataset/sequences')
parser.add_argument('--sequence_idx', type=str, default='10')

parser.add_argument('--save_gap', type=int, default=300)

args = parser.parse_args()
curr_se3 = np.eye(4)

sequence_dir = os.path.join(args.data_base_dir, args.sequence_idx, 'velodyne')
sequence_manager = Ptutils.KittiScanDirManager(sequence_dir)
scan_paths = sequence_manager.scan_fullpaths
num_frames = len(scan_paths)
num_frames_to_skip_to_show = 5
num_frames_to_save = np.floor(num_frames/num_frames_to_skip_to_show)
save_dir = "result/" + args.sequence_idx
if not os.path.exists(save_dir): os.makedirs(save_dir)
ResultSaver = PoseGraphResultSaver(init_pose= curr_se3, 
                             save_gap=args.save_gap,
                             num_frames=num_frames,
                             seq_idx=args.sequence_idx,
                             save_dir=save_dir)
for for_idx, scan_path in tqdm(enumerate(scan_paths)):
    curr_scan_pts = Ptutils.readScan(scan_path) 
    curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_points=args.num_icp_points)

    # save current node
    curr_node_idx = for_idx # make start with 0
#    SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_scan_down_pts)
    if(curr_node_idx == 0):
        prev_node_idx = curr_node_idx
        prev_scan_pts = copy.deepcopy(curr_scan_pts)
        icp_initial = np.eye(4)
        continue
    prev_scan_down_pts = Ptutils.random_sampling(prev_scan_pts, num_points=args.num_icp_points)
    odom_transform, _, _ = ICP.icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=icp_initial, max_iterations=20)

    # update the current (moved) pose 
    curr_se3 = np.matmul(curr_se3, odom_transform)
    icp_initial = odom_transform # assumption: constant velocity model (for better next ICP converges)
    prev_node_idx = curr_node_idx
    prev_scan_pts = copy.deepcopy(curr_scan_pts)
    ResultSaver.saveUnoptimizedPoseGraphResult(curr_se3, curr_node_idx) 
    if(for_idx % num_frames_to_skip_to_show == 0): 
        ResultSaver.vizCurrentTrajectory(fig_idx=1)
    