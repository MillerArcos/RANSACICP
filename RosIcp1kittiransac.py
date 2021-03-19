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
import math
import time
from nav_msgs.msg import Odometry
msg = Odometry()
from skimage.measure import LineModelND, ransac
import pyransac3d as pyrsc

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
        self.sequence_idx =  '04'
        self.for_idx=0
        self.curr_se3= np.eye(4)
        self.curr_scan_pts = []
        self.prev_scan_pts =[]
        self.curr_scan_down_pts=[]
        self.prev_scan_down_pts=[]
        self.odom_transform=[]
        self.icp_initial = np.eye(4)
        self.pose_list = np.reshape(np.eye(4), (-1, 16))
        rospy.init_node (name, anonymous = True)
        rospy.Subscriber('/velodyne_points',PointCloud2,self.read_LiDAR)
        self.pub = rospy.Publisher('/Odom', Odometry, queue_size=10)
        rospy.loginfo("Starting Node")
        rospy.spin()
        
        
    def read_LiDAR(self,data):
#        sequence_dir = os.path.join(self.data_base_dir, self.sequence_idx, 'velodyne')
#        sequence_manager = Ptutils.KittiScanDirManager(sequence_dir)
#        scan_paths = sequence_manager.scan_fullpaths
        tic = time.time()
        rate = rospy.Rate(20)
        self.num_frames = 6000
#        print('jhloi')
        # Pose Graph Manager (for back-end optimization) initialization
       
        
        # Result saver
        self.save_dir = "result/" + self.sequence_idx
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        ResultSaver = PoseGraphResultSaver(init_pose=self.curr_se3, 
                                     save_gap=self.save_gap,
                                     num_frames=self.num_frames,
                                     seq_idx=self.sequence_idx,
                                     save_dir=self.save_dir)
        
        # Scan Context Manager (for loop detection) initialization
        
        
        # for save the results as a video
        fig_idx = 1
       
        num_frames_to_skip_to_show = 5
#        num_frames_to_save = np.floor(num_frames/num_frames_to_skip_to_show)
#        with writer.saving(fig, video_name, num_frames_to_save): # this video saving part is optional
        
            # @@@ MAIN @@@: data stream
#        for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=5.0):
        
                # get current information     
        self.curr_scan_pts= ptstoxyz.pointcloud2_to_xyz_array(data,remove_nans=True)
#        print (len(self.curr_scan_pts), 'nubetotal')
#        plane1 = pyrsc.Circle()
#        best_eq, best_inliers = plane1.fit(self.curr_scan_pts, 0.01)
#        print(best_inliers)
        print(len(self.curr_scan_pts), 'origin')
        model_robust, inliers = ransac(self.curr_scan_pts, LineModelND, min_samples=10,
                               residual_threshold=1, max_trials=2)
#        print(type(inliers))
        pos_True = inliers == False
#        print(len(inliers[pos_false]) , 'outliers')
        
#        print(self.curr_scan_pts.shape)
#        curr_scan_pts = Ptutils.readScan(scan_paths) 
        
        var=[]
        for i in range(0,len(inliers)):
            if inliers[i]== False:
                var.append(i)
        self.curr_scan_pts  = self.curr_scan_pts[var[:],:] 
        print(len(self.curr_scan_pts),'ransac')
            # save current node
        curr_node_idx = self.for_idx # make start with 0
        self.curr_scan_down_pts = Ptutils.random_sampling(self.curr_scan_pts, num_points=self.num_icp_points)
        print(len(self.curr_scan_down_pts), 'random')
        if(curr_node_idx == 0):
            self.prev_node_idx = curr_node_idx
            self.prev_scan_pts = copy.deepcopy(self.curr_scan_pts)
            self.icp_initial = np.eye(4)
#            continue
    
            # calc odometry
#        print (self.prev_scan_down_pts)
        self.prev_scan_down_pts = Ptutils.random_sampling(self.prev_scan_pts, num_points=self.num_icp_points)
        self.odom_transform, _, _ = ICP.icp(self.curr_scan_down_pts, self.prev_scan_down_pts, init_pose=self.icp_initial, max_iterations=100)
#        print (self.odom_transform)
        # update the current (moved) pose 
        self.curr_se3 = np.matmul(self.curr_se3, self.odom_transform)
        
        self.icp_initial = self.odom_transform # assumption: constant velocity model (for better next ICP converges)
#        print(self.odom_transform)
        # add the odometry factor to the graph 
        
#        self.for_idx= self.for_idx+1
        # renewal the prev information 
        self.prev_node_idx = curr_node_idx
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
#        ResultSaver.saveUnoptimizedPoseGraphResult(self.curr_se3, curr_node_idx) 
        self.pose_list = np.vstack((self.pose_list, np.reshape(self.curr_se3 , (-1, 16))))
        if(curr_node_idx % self.save_gap == 0 or curr_node_idx == self.num_frames):        
            # save odometry-only poses
            filename = "pose" + self.sequence_idx + "unoptimized_" + str(int(time.time())) + ".csv"
            filename = os.path.join(self.save_dir, filename)
            np.savetxt(filename, self.pose_list, delimiter=",")
        
        
#        if(self.for_idx % num_frames_to_skip_to_show == 0): 
#            self.x = self.pose_list[:,3]
#            self.y = self.pose_list[:,7]
##            z = self.pose_list[:,11]
#    
#            fig = plt.figure(fig_idx)
#            plt.clf()
#            plt.plot(-self.y, self.x, color='blue') # kitti camera coord for clarity
#            plt.axis('equal')
#            plt.xlabel('x', labelpad=10) 
#            plt.ylabel('y', labelpad=10)
#            plt.draw()
#            plt.pause(0.01) #is necessary for the plot to update for some reason
#            ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
        self.for_idx= self.for_idx+1
#                writer.grab_frame()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = " UTM_COORDINATE "
        msg.pose.pose.position.x = -self.y[curr_node_idx-4]
        msg.pose.pose.position.y = self.x[curr_node_idx-4]
    	
         
        self.pub.publish(msg)
        rate.sleep()
        toc=time.time() 
#        print (toc-tic)
       


if __name__ == '__main__':
    node_int = laser_subs('Laser_subs')
    
        