#!/usr/bin/env python

from cmath import pi
import sys
import copy
import rospy
import moveit_commander
import geometry_msgs
from std_msgs.msg import String
import moveit_msgs.msg

import cv2
import os
from imutils.video import VideoStream
import time
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import argparse


# # ####################################################################################
# # define names of each possible ArUco tag OpenCV supports
# ARUCO_DICT = {
# 	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
# 	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
# 	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
# 	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
# 	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
# 	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
# 	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
# 	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
# 	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
# 	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
# 	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
# 	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
# 	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
# 	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
# 	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
# 	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
# 	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
# 	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
# 	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
# 	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
# 	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
# }
# # ###################################################################################



class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image  
    
    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            count = 0
            cx_dot = 0
            cy_dot = 0
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                
                if count == 8:
                    cx_dot = cx
                    cy_dot = cy
                count += 1
            # if draw:
            #     cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
            cv2.circle(image,(cx_dot,cy_dot), 15 , (255,0,255), cv2.FILLED)
            
        return lmlist
    
    

class MoveGroupPythonCalculator(object):
    
    def __init__(self):
        super(MoveGroupPythonCalculator, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_calculator", anonymous=True)

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        
        
        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.solution = 19.72
        
    def draw_0(self, scale=0.1, zscale=0.1):
        move_group = self.move_group

        waypoints = []

        wpose = move_group.get_current_pose().pose
        # Move to top right of box
        wpose.position.x += scale * 0.5
        wpose.position.y += scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        # Touch pen to paper
        wpose.position.z -= zscale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        # Stroke left
        wpose.position.x -= scale * 1.0 * 0.5
        waypoints.append(copy.deepcopy(wpose))
        # Stroke down
        wpose.position.y -= scale * 1.0
        waypoints.append(copy.deepcopy(wpose))
        # Stroke right
        wpose.position.x += scale * 1.0 * 0.5
        waypoints.append(copy.deepcopy(wpose))
        # Stroke up
        wpose.position.y += scale * 1.0
        waypoints.append(copy.deepcopy(wpose))
        # Lift off paper
        wpose.position.z += zscale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        # Move back to voxel center
        wpose.position.x += scale * 0.5
        wpose.position.y -= scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        # Plan path
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        return plan, fraction
    
    def move_left(self, scale=0.05, zscale=0.1):
        move_group = self.move_group

        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.x -= scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)

        return plan, fraction
    
    def move_left_cont(self, scale=0.02, zscale=0.1):
        move_group = self.move_group

        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.x -= scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)

        move_group.execute(plan, wait=True)
        
        

    def move_right(self, scale=0.05, zscale=0.1):
        move_group = self.move_group

        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.x += scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)

        return plan, fraction
    
    def move_up(self, scale=0.05, zscale=0.1):
        move_group = self.move_group

        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.y += scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)

        return plan, fraction
    
    def move_down(self, scale=0.05, zscale=0.1):
        move_group = self.move_group

        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.y -= scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)

        return plan, fraction
    
    def execute_plan(self, plan):
        move_group = self.move_group
        move_group.execute(plan, wait=True)
        # move_group.execute(plan)
        
    def go_to_joint_state(self):
        move_group = self.move_group

        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -2 * pi / 8
        joint_goal[2] = 0
        joint_goal[3] = -2 * pi / 4
        joint_goal[4] = 0
        joint_goal[5] = 2 * pi / 6 

        move_group.go(joint_goal, wait=True)
        move_group.stop()
        
    def go_to_initial_goal(self):
        move_group = self.move_group
        
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = -1.117
        joint_goal[1] = -2.077
        joint_goal[2] = -1.588
        joint_goal[3] = -1.065
        joint_goal[4] = 1.012
        joint_goal[5] = -5.428


        # pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.x = 1.0
        # pose_goal.orientation.y = 0.0
        # pose_goal.orientation.z = 0.0
        # pose_goal.orientation.w = 0.0
        # # pose_goal.orientation.x = 0.924
        # # pose_goal.orientation.y = 0.0
        # # pose_goal.orientation.z = 0.0
        # # pose_goal.orientation.w = 0.383
        # # pose_goal.position.x = -0.4
        # # pose_goal.position.y = 0.2
        # # pose_goal.position.z = 0.2
        # pose_goal.position.x = -0.05
        # pose_goal.position.y = 0.4
        # pose_goal.position.z = 0.2

        # move_group.set_pose_target(pose_goal)
        
        move_group.set_pose_target(joint_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_left_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.x -= control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_right_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.x += control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_up_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.y += control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_down_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.y -= control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_tall_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.z -= control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_short_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.z += control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def scan_for_objects(self, pipeline, target_number):
    # def scan_for_objects(self, pipeline):
        # needs to be completed
        continue_flag = True
        triggerTime = time.time()
        while continue_flag:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            
            start = time.time()
            if (start-triggerTime >= 0.1):
                colfil1 = np.array([0, 0, 100])
                colfil2 = np.array([50, 50, 255]) #currently red
                mask = cv2.inRange(color_image, colfil1, colfil2)
                color_mask = cv2.bitwise_and(color_image,color_image, mask= mask)
                cv2.addWeighted(color_mask, 1, color_image, 1, 0, color_image)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                lmList = [300, 250]
                
                # All contours detected
                bounded_contours = []
                for pic, contour in enumerate(contours): # extend this to create matrix or lists, track initial positon to estimate location in scene and store objects
                    # if counter == 0:
                    area = cv2.contourArea(contour)
                    if (area > 500):
                        x, y, w, h = cv2.boundingRect(contour)
                        # cv2.rectangle(qcolor_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        lmList = [int(x + w/2), int(y + h/2)]
                        
                        bounded_contours.append([x, y, w, h])
                
                final_contours = []
                
                # Creating contour check dictionary
                contour_check = {}     
                for i in range(len(bounded_contours)):
                    contour_check[i] = True
                
                # Looking for overlaps in contours
                for i in range(len(bounded_contours)):
                    if contour_check[i] == True:
                        current_box = bounded_contours[i]
                        for j in range(i + 1, len(bounded_contours)):
                            if (((bounded_contours[j][0] < current_box[0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[j][0] < current_box[0] + current_box[2] < bounded_contours[j][0] + bounded_contours[j][2])) and ((bounded_contours[j][0] < current_box[0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[j][1] < current_box[1] + current_box[3] < bounded_contours[j][1] + bounded_contours[j][3]))):
                                if bounded_contours[j][0] < current_box[0]:
                                    current_box[0] = bounded_contours[j][0]
                                if (bounded_contours[j][0] + bounded_contours[j][2]) > (current_box[0] + current_box[2]):
                                    current_box[2] = bounded_contours[j][0] + bounded_contours[j][2] - current_box[0]
                                if bounded_contours[j][1] < current_box[1]:
                                    current_box[1] = bounded_contours[j][1]
                                if (bounded_contours[j][1] + bounded_contours[j][3]) > (current_box[1] + current_box[3]):
                                    current_box[3] = bounded_contours[j][1] + bounded_contours[j][3] - current_box[1]
                                contour_check[j] = False
                        final_contours.append(current_box)
                
                print(len(final_contours))
                
                # remove any encircled contours
                final_contours2 = []
                # try:
                for i in range(len(final_contours)):
                    for j in range(len(final_contours)):
                        if not (final_contours[i][0] < final_contours[j][0] and (final_contours[i][0] + final_contours[i][2]) > (final_contours[j][0] + final_contours[j][2]) and (final_contours[i][1] < final_contours[j][1]) and (final_contours[i][1] + final_contours[i][3]) > (final_contours[j][1] + final_contours[j][3])):
                            final_contours2.append(final_contours[j])
                # except ValueError:
                #     pass
                final_contours = final_contours2
                print(final_contours2)
                
                # try:
                #     if len(final_contours) > target_number:
                #         largest_areas = np.zeros(target_number)
                #         largest_contours = [[]] * target_number
                #         for contour in final_contours:
                #             if (contour[2] * contour[3]) > largest_areas[2]:
                #                 largest_areas[2] = contour[2] * contour[3]
                #                 largest_contours[2] = contour
                #             elif (contour[2] * contour[3]) > largest_areas[1]:
                #                 largest_areas[1] = contour[2] * contour[3]
                #                 largest_contours[1] = contour
                #             elif (contour[2] * contour[3]) > largest_areas[0]:
                #                 largest_areas[0] = contour[2] * contour[3]
                #                 largest_contours[0] = contour
                #     else:
                #         largest_contours = final_contours
                        
                # except IndexError:
                #     largest_contours = final_contours
                
                largest_contours = final_contours2

                # Selecting three largest regions (put in if statement to see if we have more than three contours)
                largest_areas = [0, 0, 0]
                largest_contours = [[], [], []]
                for contour in bounded_contours:
                    if (contour[2] * contour[3]) > largest_areas[2]:
                        largest_areas[2] = contour[2] * contour[3]
                        largest_contours[2] = contour
                    elif (contour[2] * contour[3]) > largest_areas[1]:
                        largest_areas[1] = contour[2] * contour[3]
                        largest_contours[1] = contour
                    elif (contour[2] * contour[3]) > largest_areas[0]:
                        largest_areas[0] = contour[2] * contour[3]
                        largest_contours[0] = contour
                
                print(largest_contours)
                
                # Storing centers of three largest contours
                # try: 
                #     for contour in largest_contours:
                #         cv2.rectangle(color_image, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]), (255, 0, 0), 2)
                #     stored_positions = []
                #     for contour in largest_contours:
                #         stored_positions.append([contour[0] + contour[2]//2, contour[1] + contour[3]//2])
                # except IndexError:
                #     pass
                
                stored_positions = []
                for contour in largest_contours:
                    try:
                        cv2.rectangle(color_image, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]), (255, 0, 0), 2)
                        stored_positions.append([contour[0] + contour[2]//2, contour[1] + contour[3]//2])
                    except IndexError:
                        pass
            
                    
                # try:
                #     for i in range(len(bounded_contours)):
                #         for j in range(len(bounded_contours)):
                #             if (bounded_contours[i][0] < bounded_contours[j][0] and (bounded_contours[i][0] + bounded_contours[i][2]) > (bounded_contours[j][0] + bounded_contours[j][2]) and (bounded_contours[i][1] < bounded_contours[j][1]) and (bounded_contours[i][1] + bounded_contours[i][3]) > (bounded_contours[j][1] + bounded_contours[j][3])):
                #                 bounded_contours.remove(contours[j])
                    
                    # final_contours = []            
                    # for i in range(len(bounded_contours)):
                    #     change_flag = False
                    #     for j in range(i + 1, len(bounded_contours)):
                    #         if (bounded_contours[j][0] < bounded_contours[i][0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[i][0] < bounded_contours[j][0] < bounded_contours[i][0] + bounded_contours[i][2]) or (bounded_contours[j][1] < bounded_contours[i][1] < bounded_contours[j][1] + bounded_contours[j][3]) or (bounded_contours[i][1] < bounded_contours[j][1] < bounded_contours[i][1] + bounded_contours[i][3]):
                    #             new_x = min(bounded_contours[i][0], bounded_contours[j][0])
                    #             new_y = min(bounded_contours[i][1], bounded_contours[j][1])
                    #             new_x_h = max(bounded_contours[i][0] + bounded_contours[i][2], bounded_contours[j][0] + bounded_contours[j][2])
                    #             new_y_h = max(bounded_contours[i][1] + bounded_contours[i][3], bounded_contours[j][1] + bounded_contours[j][3])
                    #             final_contours.append([new_x, new_y, new_x_h - new_x, new_y_h - new_y])
                    #             change_flag = True
                    #             break
                    #     if change_flag == False:
                    #         final_contours.append(bounded_contours[i])
                            
                    # print(len(final_contours))
                            
                    
                    # for i in range(len(bounded_contours)):
                    #     for j in range(len(bounded_contours)):
                    #         if (bounded_contours[j][0] < bounded_contours[i][0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[i][0] < bounded_contours[j][0] < bounded_contours[i][0] + bounded_contours[i][2]) or (bounded_contours[j][1] < bounded_contours[i][1] < bounded_contours[j][1] + bounded_contours[j][3]) or (bounded_contours[i][1] < bounded_contours[j][1] < bounded_contours[i][1] + bounded_contours[i][3]):
                    #             new_x = min(bounded_contours[i][0], bounded_contours[j][0])
                    #             new_y = min(bounded_contours[i][1], bounded_contours[j][1])
                    #             new_x_h = max(bounded_contours[i][0] + bounded_contours[i][2], bounded_contours[j][0] + bounded_contours[j][2])
                    #             new_y_h = max(bounded_contours[i][1] + bounded_contours[i][3], bounded_contours[j][1] + bounded_contours[j][3])
                    #             bounded_contours[i] = [new_x, new_y, new_x_h - new_x, new_y_h - new_y]
                    #             bounded_contours.remove(contours[j])
                                
                    # print(len(bounded_contours))
                                
                    # print(len(bounded_contours))
                    
                    # set_flag = False
                    # # if (len(bounded_contours) > target_number):
                    # final_contours = []
                    # for i in range(len(bounded_contours)):
                    #     change_flag = False
                    #     for j in range(len(bounded_contours)):
                    #         if (bounded_contours[j][0] < bounded_contours[i][0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[i][0] < bounded_contours[j][0] < bounded_contours[i][0] + bounded_contours[i][2]) or (bounded_contours[j][1] < bounded_contours[i][1] < bounded_contours[j][1] + bounded_contours[j][3]) or (bounded_contours[i][1] < bounded_contours[j][1] < bounded_contours[i][1] + bounded_contours[i][3]):
                    #             new_x = min(bounded_contours[i][0], bounded_contours[j][0])
                    #             new_y = min(bounded_contours[i][1], bounded_contours[j][1])
                    #             new_x_h = max(bounded_contours[i][0] + bounded_contours[i][2], bounded_contours[j][0] + bounded_contours[j][2])
                    #             new_y_h = max(bounded_contours[i][1] + bounded_contours[i][3], bounded_contours[j][1] + bounded_contours[j][3])
                    #             # temp1 = bounded_contours[i]
                    #             # temp2 = bounded_contours[j]
                    #             # bounded_contours.remove(temp1)
                    #             # bounded_contours.remove(temp2)
                    #             # bounded_contours.append([new_x, new_y, new_x_h - new_x, new_y_h - new_y])
                    #             final_contours.append([new_x, new_y, new_x_h - new_x, new_y_h - new_y])
                    #             set_flag = True
                    #     if change_flag == False:
                    #         final_contours.append(bounded_contours[i])
                    
                    # end = len(bounded_contours)
                    # print(end)
                    # i = 0
                    # j = 0          
                    # while i < end:
                    #     print('within i')
                    #     while j < end:
                    #         print('within j')
                    #         if i == j:
                    #             j = j + 1
                    #         if (bounded_contours[j][0] < bounded_contours[i][0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[i][0] < bounded_contours[j][0] < bounded_contours[i][0] + bounded_contours[i][2]) or (bounded_contours[j][1] < bounded_contours[i][1] < bounded_contours[j][1] + bounded_contours[j][3]) or (bounded_contours[i][1] < bounded_contours[j][1] < bounded_contours[i][1] + bounded_contours[i][3]):
                    #             new_x = min(bounded_contours[i][0], bounded_contours[j][0])
                    #             new_y = min(bounded_contours[i][1], bounded_contours[j][1])
                    #             new_x_h = max(bounded_contours[i][0] + bounded_contours[i][2], bounded_contours[j][0] + bounded_contours[j][2])
                    #             new_y_h = max(bounded_contours[i][1] + bounded_contours[i][3], bounded_contours[j][1] + bounded_contours[j][3])
                    #             bounded_contours.insert(i, [new_x, new_y, new_x_h - new_x, new_y_h - new_y])
                    #             temp1 = bounded_contours[i]
                    #             temp2 = bounded_contours[j]
                    #             bounded_contours.remove(temp1)
                    #             bounded_contours.remove(temp2)
                    #         else:
                    #             j = j + 1
                    #         end = len(bounded_contours)    
                    #     i = i + 1
                    #     j = 0
                    #     end = len(bounded_contours)
                            
                                    
                    # if set_flag == True:
                    #     final_contours = bounded_contours           
                    #     print(len(bounded_contours))
                    #     print('FINAL CONTOURS FOUND')
                    #     continue_flag = False
                    
                # for contour in bounded_contours:
                try:
                    for contour in largest_contours:
                        cv2.rectangle(color_image, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]), (255, 0, 0), 2)
                except IndexError:
                    pass
                        
                bounds = [295, 245, 305, 255]
                depth_limits = [-0.01, 0.01]
                cv2.rectangle(color_image, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 255, 0), 2)

                cv2.imshow("Video",color_image)
                cv2.waitKey(1)
                
                triggerTime = time.time()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        return stored_positions
    
    # def move_to_object(self, object):
    def move_to_object(self, pipeline):
        at_object = False
        triggerTime = time.time()
        while not at_object:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            
            depth_target = 0.20
            avgdepth = depth_target
            
            start = time.time()
            if (start-triggerTime >= 0.1):
                colfil1 = np.array([0, 0, 100])
                colfil2 = np.array([50, 50, 255]) #currently red
                mask = cv2.inRange(color_image, colfil1, colfil2)
                color_mask = cv2.bitwise_and(color_image,color_image, mask= mask)
                cv2.addWeighted(color_mask, 1, color_image, 1, 0, color_image)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                lmList = [300, 250]
                
                for pic, contour in enumerate(contours): # extend this to create matrix or lists, track initial positon to estimate location in scene and store objects
                    # if counter == 0:
                    area = cv2.contourArea(contour)
                    if (area > 500):
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        lmList = [int(x + w/2), int(y + h/2)]
                        
                bounds = [295, 245, 305, 255]
                depth_limits = [-0.01, 0.01]
                cv2.rectangle(color_image, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 255, 0), 2)

                cv2.imshow("Video",color_image)
                cv2.waitKey(1)
                
                try:
                    cx_dot = lmList[0]
                    cy_dot = lmList[1]
                    depth = 0
                    count = 0
                    
                    try:
                        for i in range(lmList[0] - 25, lmList[0] + 25):
                            for j in range(lmList[1] - 25, lmList[1] + 25):
                                depth = depth + depth_frame.get_distance(i,j)
                                count = count+1
                        avgdepth = depth/count
                        cv2.rectangle(color_image, (lmList[0] - 10, lmList[0] + 10), (lmList[1] - 10, lmList[1] + 10), (0, 0, 255), 2)
                        if avgdepth < .1:
                            avgdepth = depth_target
                        
                        print(avgdepth)
                    
                    except RuntimeError:
                        pass
                        
                except IndexError:
                    cx_dot = bounds[0] + (bounds[2] - bounds[0]) / 2
                    cy_dot = bounds[1] + (bounds[3] - bounds[1]) / 2
                    
                if cx_dot < bounds[0]:
                    print("Out of bounds (left)")
                    control = abs(cx_dot - bounds[0]) * 0.02/100
                    # # cartesian_plan, fraction = tutorial.move_left()
                    # # tutorial.execute_plan(cartesian_plan)
                    self.move_left_goal(control)
                if cy_dot < bounds[1]:
                    print("Out of bounds (up)")
                    control = abs(cy_dot - bounds[1]) * 0.02/100
                    # cartesian_plan, fraction = tutorial.move_up()
                    # tutorial.execute_plan(cartesian_plan)
                    self.move_up_goal(control)
                if cx_dot > bounds[2]:
                    print("Out of bounds (right)")
                    control = abs(cx_dot - bounds[2]) * 0.02/100
                    # cartesian_plan, fraction = tutorial.move_right()
                    # tutorial.execute_plan(cartesian_plan)
                    self.move_right_goal(control)
                if cy_dot > bounds[3]:
                    print("Out of bounds (down)")
                    control = abs(cy_dot - bounds[3]) * 0.02/100
                    # cartesian_plan, fraction = tutorial.move_down()
                    # tutorial.execute_plan(cartesian_plan)
                    self.move_down_goal(control)
                # Vertical control turned off due to lack of reliability
                if avgdepth > depth_target + depth_limits[1]:
                    print("Out of bounds (tall)")
                    print(avgdepth)
                    control = abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 if abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 < 0.025 else 0.025
                    print(control)
                    # self.move_tall_goal(control)
                if avgdepth < depth_target + depth_limits[0]:
                    print("Out of bounds (short)")
                    print(avgdepth)
                    control = abs(avgdepth - (depth_target - depth_limits[0])) * 10/100 if abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 < 0.025 else 0.025
                    print(control)
                    # self.move_short_goal(control)
                    
                triggerTime = time.time()
                
                if bounds[0] <= cx_dot <= bounds[2] and bounds[1] <= cy_dot <= bounds[3]:
                    at_object = True
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                    
    
def main():
    try:
        print("")
        print("----------------------------------------------------------")
        print("")
        input("============ Press `Enter` to initialize ...")
        
        # Initialize Tutorial Object
        tutorial = MoveGroupPythonCalculator()
        
        # Hand Tracking Stuff
        # cap = cv2.VideoCapture(0)
        
        # #########################################################################
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        #print(device) --> pyrealsense2.device: Intel RealSense D435 (S/N: 922612071311  FW: 05.12.06.00  on USB3.2
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        #print(device_product_line) --> D400

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
        # #########################################################################
        
        # Initialize Hand Tracker Object
        # tracker = handTracker()

        input("============ Press `Enter` to assume initial pose goal ...")
        tutorial.go_to_initial_goal()
        
        # input()
        # tutorial.move_tall_goal(0.063)
        
        # #####################################################################
        # # input()
        # tutorial.move_short_goal(0.15)
        # #####################################################################
        
        input("============ Press `Enter` to begin control ...")
        tutorial.scan_for_objects(pipeline, 3)
        
        # #####################################################################
        # FROM PICK DEMO
        # tutorial.move_to_object(pipeline)
        # print('Reached Object!')
        # # input()
        
        # tutorial.move_left_goal(0.08) # combine into single 'pick' function
        # tutorial.move_up_goal(0.028)
        # # input()
        # tutorial.move_tall_goal(0.15)
        # tutorial.move_tall_goal(0.063)
        # # input()
        # tutorial.move_short_goal(0.2)
        # #####################################################################
        
        # input()
        # # Hand Tracking Stuff
        # triggerTime = time.time()
        # while True:
        #     # #####################################################################
        #     # Wait for a coherent pair of frames: depth and color
        #     frames = pipeline.wait_for_frames()
        #     depth_frame = frames.get_depth_frame()
        #     color_frame = frames.get_color_frame()
            
        #     # ############################################################
        #     # hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        #     # red_lower = np.array([136, 87, 111], np.uint8)
        #     # red_upper = np.array([180, 255, 255], np.uint8)
        #     # red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
        #     # ############################################################

        #     if not depth_frame or not color_frame:
        #         continue

        #     # Convert images to numpy arrays
        #     depth_image = np.asanyarray(depth_frame.get_data())
        #     color_image = np.asanyarray(color_frame.get_data())
            
        #     # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #     depth_colormap_dim = depth_colormap.shape
        #     color_colormap_dim = color_image.shape
            
        #     depth_target = 0.20
        #     avgdepth = depth_target
        #     # ####################################################################
            
        #     start = time.time()
        #     if (start-triggerTime >= 0.1):
                
        #         # ###########################################################
        #         colfil1 = np.array([0, 0, 100])
        #         colfil2 = np.array([50, 50, 255]) #currently red
        #         mask = cv2.inRange(color_image, colfil1, colfil2)
        #         color_mask = cv2.bitwise_and(color_image,color_image, mask= mask)
        #         cv2.addWeighted(color_mask, 1, color_image, 1, 0, color_image)
        #         contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
        #         # contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
        #         # bound_rect = cv2.boundingRect(list(contour))
        #         # pt1 = (bound_rect[0], bound_rect[1])
        #         # pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
        #         # cv2.rectangle(color_image, pt1, pt2, (0, 0, 255), 2)
                
        #         lmList = [300, 250]
                
        #         # print(color_mask)
        #         # counter = 0
        #         for pic, contour in enumerate(contours):
        #             # if counter == 0:
        #             area = cv2.contourArea(contour)
        #             if (area > 500):
        #                 x, y, w, h = cv2.boundingRect(contour)
        #                 cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #                 lmList = [int(x + w/2), int(y + h/2)]
        #         #     counter += 1
        #         # ###########################################################
                
        #         # Update with information from hand tracker
        #         # color_image = tracker.handsFinder(color_image)
        #         # lmList = tracker.positionFinder(color_image)

        #         # Bounds for central rectangular prism
        #         # bounds = [100, 100, 500, 400]
        #         # bounds = [200, 150, 400, 350]
        #         # depth_limits = [-0.05, 0.05]
        #         bounds = [250, 200, 350, 300]
        #         depth_limits = [-0.01, 0.01]
        #         cv2.rectangle(color_image, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 255, 0), 2)

        #         cv2.imshow("Video",color_image)
        #         cv2.waitKey(1)
                
        #         try:
        #             cx_dot = lmList[0]
        #             cy_dot = lmList[1]
        #             depth = 0
        #             count = 0
                    
        #             try:
        #                 for i in range(lmList[0] - 25, lmList[0] + 25):
        #                     for j in range(lmList[1] - 25, lmList[1] + 25):
        #                         depth = depth + depth_frame.get_distance(i,j)
        #                         count = count+1
        #                 avgdepth = depth/count
        #                 cv2.rectangle(color_image, (lmList[0] - 10, lmList[0] + 10), (lmList[1] - 10, lmList[1] + 10), (0, 0, 255), 2)
        #                 if avgdepth < .1:
        #                     avgdepth = depth_target
                        
        #                 print(avgdepth)
                    
        #             except RuntimeError:
        #                 pass
                        
        #         except IndexError:
        #             cx_dot = bounds[0] + (bounds[2] - bounds[0]) / 2
        #             cy_dot = bounds[1] + (bounds[3] - bounds[1]) / 2
                
        #         # end = time.time()
        #         # if (end - start) > 0.:
        #         #     return
        #         # else:
        #         # Check Breaches
        #         if cx_dot < bounds[0]:
        #             print("Out of bounds (left)")
        #             control = abs(cx_dot - bounds[0]) * 0.02/100
        #             # # cartesian_plan, fraction = tutorial.move_left()
        #             # # tutorial.execute_plan(cartesian_plan)
        #             tutorial.move_left_goal(control)
        #         if cy_dot < bounds[1]:
        #             print("Out of bounds (up)")
        #             control = abs(cy_dot - bounds[1]) * 0.02/100
        #             # cartesian_plan, fraction = tutorial.move_up()
        #             # tutorial.execute_plan(cartesian_plan)
        #             tutorial.move_up_goal(control)
        #         if cx_dot > bounds[2]:
        #             print("Out of bounds (right)")
        #             control = abs(cx_dot - bounds[2]) * 0.02/100
        #             # cartesian_plan, fraction = tutorial.move_right()
        #             # tutorial.execute_plan(cartesian_plan)
        #             tutorial.move_right_goal(control)
        #         if cy_dot > bounds[3]:
        #             print("Out of bounds (down)")
        #             control = abs(cy_dot - bounds[3]) * 0.02/100
        #             # cartesian_plan, fraction = tutorial.move_down()
        #             # tutorial.execute_plan(cartesian_plan)
        #             tutorial.move_down_goal(control)
        #         # Vertical control turned off due to lack of reliability
        #         if avgdepth > depth_target + depth_limits[1]:
        #             print("Out of bounds (tall)")
        #             print(avgdepth)
        #             control = abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 if abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 < 0.025 else 0.025
        #             print(control)
        #             # tutorial.move_tall_goal(control)
        #         if avgdepth < depth_target + depth_limits[0]:
        #             print("Out of bounds (short)")
        #             print(avgdepth)
        #             control = abs(avgdepth - (depth_target - depth_limits[0])) * 10/100 if abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 < 0.025 else 0.025
        #             print(control)
        #             # tutorial.move_short_goal(control)
                    
        #         triggerTime = time.time()
        #        
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break
            
        cv2.destroyAllWindows() 
        
        input("============ Press `Enter` to terminate ...")
        moveit_commander.roscpp_shutdown()

        # tutorial.go_to_initial_goal()
        
        print("============ Python calculator complete!")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()