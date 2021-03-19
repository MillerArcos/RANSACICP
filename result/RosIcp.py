#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import random
import copy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import tila1 as ptstoxyz
import os
import sys
import csv

import time



from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
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

class laser_subs(object): #Para crear clase
 
    def __init__(self, name):
     
        self.num_icp_points = 5000
        self.num_rings = 20
        self.num_sectors = 60 
        self.num_candidates = 10
        self.try_gap_loop_detection = 10
        self.loop_threshold = 0.11
        self.save_gap = 300
        self.data_base_dir= '/home/miller/dataset/sequences'
        self.sequence_idx =  '30'
        self.for_idx=0
        self.curr_scan_pts = []
        self.prev_scan_pts =[]
        self.curr_scan_down_pts=[]
        self.prev_scan_down_pts=[]
        self.odom_transform=[]
        self.icp_initial = np.eye(4)
        rospy.init_node (name, anonymous = True)
        rospy.Subscriber('/quanergy/points',PointCloud2,self.read_LiDAR)
        rospy.loginfo("Starting Node")
        rospy.spin()
        
    def read_LiDAR(self,data):
#        sequence_dir = os.path.join(self.data_base_dir, self.sequence_idx, 'velodyne')
#        sequence_manager = Ptutils.KittiScanDirManager(sequence_dir)
#        scan_paths = sequence_manager.scan_fullpaths
        print (self.for_idx)
        num_frames = 6000
        print('jhloi')
        # Pose Graph Manager (for back-end optimization) initialization
        PGM = PoseGraphManager()
        PGM.addPriorFactor()
        
        # Result saver
        save_dir = "result/" + self.sequence_idx
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3, 
                                     save_gap=self.save_gap,
                                     num_frames=num_frames,
                                     seq_idx=self.sequence_idx,
                                     save_dir=save_dir)
        
        # Scan Context Manager (for loop detection) initialization
        SCM = ScanContextManager(shape=[self.num_rings, self.num_sectors], 
                                                num_candidates=self.num_candidates, 
                                                threshold=self.loop_threshold)
        
        # for save the results as a video
        fig_idx = 1
        fig = plt.figure(fig_idx)
#        writer = FFMpegWriter(fps=15)
#        video_name = self.sequence_idx + "_" + str(self.num_icp_points) + ".mp4"
        num_frames_to_skip_to_show = 5
#        num_frames_to_save = np.floor(num_frames/num_frames_to_skip_to_show)
#        with writer.saving(fig, video_name, num_frames_to_save): # this video saving part is optional
        
            # @@@ MAIN @@@: data stream
#        for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=5.0):
        
                # get current information     
        self.curr_scan_pts= ptstoxyz.pointcloud2_to_xyz_array(data,remove_nans=True)
#        curr_scan_pts = Ptutils.readScan(scan_paths) 
        self.curr_scan_down_pts = Ptutils.random_sampling(self.curr_scan_pts, num_points=self.num_icp_points)
    
            # save current node
        PGM.curr_node_idx = self.for_idx # make start with 0
        SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=self.curr_scan_down_pts)
        if(PGM.curr_node_idx == 0):
            PGM.prev_node_idx = PGM.curr_node_idx
            self.prev_scan_pts = copy.deepcopy(self.curr_scan_pts)
            self.icp_initial = np.eye(4)
#            continue
    
            # calc odometry
        print (self.prev_scan_down_pts)
        self.prev_scan_down_pts = Ptutils.random_sampling(self.prev_scan_pts, num_points=self.num_icp_points)
        self.odom_transform, _, _ = ICP.icp(self.curr_scan_down_pts, self.prev_scan_down_pts, init_pose=self.icp_initial, max_iterations=20)
        print (self.odom_transform)
        # update the current (moved) pose 
        PGM.curr_se3 = np.matmul(PGM.curr_se3, self.odom_transform)
        print (PGM.curr_se3)
        self.icp_initial = self.odom_transform # assumption: constant velocity model (for better next ICP converges)

        # add the odometry factor to the graph 
        PGM.addOdometryFactor(self.odom_transform)
        self.for_idx= self.for_idx+1
        # renewal the prev information 
        PGM.prev_node_idx = PGM.curr_node_idx
        self.prev_scan_pts = copy.deepcopy(self.curr_scan_pts)

        # loop detection and optimize the graph 
#        if(PGM.curr_node_idx > 1 and PGM.curr_node_idx % self.try_gap_loop_detection == 0): 
#            # 1/ loop detection 
#            loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
#            if(loop_idx == None): # NOT FOUND
#                pass
#            else:
#                print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
#                # 2-1/ add the loop factor 
#                loop_scan_down_pts = SCM.getPtcloud(loop_idx)
#                loop_transform, _, _ = ICP.icp(curr_scan_down_pts, loop_scan_down_pts, init_pose=yawdeg2se3(yaw_diff_deg), max_iterations=20)
#                PGM.addLoopFactor(loop_transform, loop_idx)
#
#                # 2-2/ graph optimization 
#                PGM.optimizePoseGraph()
#
#                # 2-2/ save optimized poses
#                ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

        # save the ICP odometry pose result (no loop closure)
        ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx) 
        if(self.for_idx % num_frames_to_skip_to_show == 0): 
            ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
        self.for_idx= self.for_idx+1
#                writer.grab_frame()
       


if __name__ == '__main__':
    node_int = laser_subs('Laser_subs')
    
        