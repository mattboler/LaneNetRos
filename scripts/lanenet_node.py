#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# @Author  : Luo Yao
# @Modified  : AdamShan
# @Original site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_node.py


import time
import math
import tensorflow as tf
import numpy as np
import cv2

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from config import global_config

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from lane_detector.msg import Lane_Image


CFG = global_config.cfg

def coords_to_h_transform(coords):
    cols = coords.shape[1]
    ones = np.ones((1, cols))
    coords_h = np.vstack((coords, ones))
    return coords_h

def coords_h_to_coords_transform(coords_h):
    coords = np.divide(coords_h, coords_h[-1,:])
    return coords[:-1,:]


class lanenet_detector():
    def __init__(self):
        # ROS Stuff
        self.image_topic = rospy.get_param('~image_topic')
        self.output_image = rospy.get_param('~output_image')
        self.output_lane = rospy.get_param('~output_lane')
        self.weight_path = rospy.get_param('~weight_path')
        self.use_gpu = rospy.get_param('~use_gpu')
        self.lane_image_topic = rospy.get_param('~lane_image_topic')

        self.init_lanenet()
        self.bridge = CvBridge()
        # Buffer size 10mb?
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=1, buff_size = 100000000)
        self.pub_image = rospy.Publisher(self.output_image, Image, queue_size=1)
        self.pub_laneimage = rospy.Publisher(self.lane_image_topic, Lane_Image, queue_size=1)
        self.pub_nav = rospy.Publisher('lane_detector/waypoints', Path)
        
        # Camera Stuff
        self.R = np.asarray([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]], np.float32)
        self.t = np.asarray([0, 0, -1.518], np.float32)
        self.T = np.eye(4)
        self.T[0:3,0:3] = self.R
        self.T[0:3,3] = self.t
        self.K = np.asarray([[617.2716, 0, 327.2818],
                        [0, 617.1263, 245.0939],
                        [0, 0, 1]], np.float32)
        self.lane_width = 3.8 #m

        ground_plane_normal_w = np.array([[0],[0],[1]])
        self.normal_c = np.dot(self.R, ground_plane_normal_w)
        origin_w = np.asarray([[0],[0],[0]], np.float32)
        origin_c = self.map_w_to_c(origin_w)
        self.dot_const_c = np.dot(self.normal_c.T, origin_c)

        self.top_limit  = 275
        self.bottom_limit = 415
    
    def init_lanenet(self):
        '''
        initlize the tensorflow model
        '''

        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)
        net = lanenet.LaneNet(phase=phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        # self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()

        saver = tf.train.Saver()
        # Set sess configuration
        if self.use_gpu:
            print("Running in GPU mode")
            sess_config = tf.ConfigProto(device_count={'GPU': 1})
            sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
            sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
            sess_config.gpu_options.allocator_type = 'BFC'
            self.sess = tf.Session(config=sess_config)
        else:
            print("Running in CPU mode")
            self.sess = tf.Session()

        saver.restore(sess=self.sess, save_path=self.weight_path)
        print("Restored session")
        

    
    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", cv_image)
        # cv2.waitKey(0)
        original_img = cv_image.copy()
        resized_image = self.preprocessing(cv_image)
        # Here we have a grayscale image with lanes
        mask_image = self.inference_net(resized_image, original_img)
        print (np.sum(mask_image))

        # Begin spline fitting!
        # 1: Figure out which color is left and which color is right
        """
        Split image in half; find centroids of every color in each half.
        Left lane will have centroid furthest right in left half
        Right lane will have centroid furthest left in right half
        """
        height, width = mask_image.shape
        width_cutoff = width // 2
        left_img = mask_image[:, :width_cutoff]
        right_img = mask_image[:, width_cutoff:]

        left_vals = left_img[left_img > 0]
        if left_vals.size > 100:
            left_id = int(np.median(left_vals))
            left_count = left_vals[left_vals == left_id].size
        else:
            left_id = -1
            left_count = 0


        right_vals = right_img[right_img > 0]
        if right_vals.size > 100:
            right_id = int(np.median(right_vals))
            right_count = right_vals[right_vals == right_id].size
        else:
            right_id = -1
            right_count = 0
        
        if left_count == 0 and right_count == 0:
            state = "TRACKING_LOST"
            return
        elif left_count == 0:
            state = "RIGHT_ONLY"
        elif right_count == 0:
            state = "LEFT_ONLY"
        else:
            state = "BOTH_LANES"
        
        print state
        
        # Build data for polyfit
        if state == "RIGHT_ONLY" or state == "BOTH_LANES":
            right_ind = np.argwhere(mask_image == right_id)
            right_yvals = right_ind[:,0]
            right_xvals = right_ind[:,1]
            right_fit = np.polyfit(right_yvals, right_xvals, 2)

        if state == "LEFT_ONLY" or state == "BOTH_LANES":
            left_ind = np.argwhere(mask_image == left_id)
            left_yvals = left_ind[:,0]
            left_xvals = left_ind[:,1]
            left_fit = np.polyfit(left_yvals, left_xvals, 2)

        
        if state == "RIGHT_ONLY":
            # Only look for right lane, guess at center by avg lane width
            plot_y = np.linspace(self.bottom_limit, self.top_limit, 50)
            right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
            right_points = np.vstack([right_fit_x, plot_y])
            right_world_pts = self.pix_to_world(right_points)
            right_offset = np.array([[0], [self.lane_width/2]])
            right_offset_arr = np.tile(right_offset, 50)
            world_pts = right_world_pts + right_offset_arr
        elif state == "LEFT_ONLY":
            # Only look for left lane, guess at center by avg lane width
            plot_y = np.linspace(self.bottom_limit, self.top_limit, 50)
            left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
            left_points = np.vstack([left_fit_x, plot_y])
            left_world_pts = self.pix_to_world(left_points)
            left_offset = np.array([[0], [-self.lane_width/2]])
            left_offset_arr = np.tile(left_offset, 50)
            world_pts = left_world_pts + left_offset_arr
        elif state == "BOTH_LANES": 
            # Fit both, interpolate for center
            center_fit = (left_fit + right_fit) / 2
            plot_y = np.linspace(self.bottom_limit, self.top_limit, 50)
            center_fit_x = center_fit[0]*plot_y**2 + center_fit[1]*plot_y + center_fit[2]
            center_points = np.vstack([center_fit_x, plot_y])
            world_pts = self.pix_to_world(center_points)
        
        #print world_pts
        n_points = world_pts.shape[1]
        n_out = n_points - 1

        angles_out = np.zeros((n_out))
        
        path = Path()
        path.header = data.header

        for i in range(n_out):
            x1 = world_pts[0,i]
            y1 = world_pts[1,i]
            x2 = world_pts[0,i+1]
            y2 = world_pts[1,i+1]
            dx = x2 - x1
            dy = y2 - y1
            theta = np.arctan2(dy, dx)
            angles_out[i] = theta
            x = float(x1)
            y = float(y1)
            w = np.cos(theta/2)
            z = np.sin(theta/2)
            P = PoseStamped()
            p = Pose()
            #p.position.x = x1
            #p.position.y = y1
            p.position.x = i
            p.position.y = i
            p.position.z = 0
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = z
            p.orientation.w = w
            P.pose = p
            P.header = data.header
            path.poses.append(P)
            #print P
        
        #print path
        
        self.pub_nav.publish(path)
        #print path

        # convert to color and publish
        color_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        out_img_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        self.pub_image.publish(out_img_msg)
        
    def preprocessing(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", image)
        # cv2.waitKey(1)
        return image

    def inference_net(self, img, original_img):
        binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                        feed_dict={self.input_tensor: [img]})

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=original_img
        )
        # mask_image = postprocess_result['mask_image']
        mask_image = postprocess_result
        mask_image = cv2.resize(mask_image, (original_img.shape[1],
                                                original_img.shape[0]),interpolation=cv2.INTER_LINEAR)
        # Don't want overlaid; want raw pixel outs
        # mask_image = cv2.addWeighted(original_img, 0.6, mask_image, 5.0, 0)
        return mask_image


    def minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr
    
    def map_w_to_c(self, coords):
        return np.dot(self.R, coords_h_to_coords_transform(
            np.dot(self.T, coords_to_h_transform(coords))))
    
    def pix_to_world(self, pix_coords):
        pix_coords_h = coords_to_h_transform(pix_coords)
        pix_norm = np.dot(np.linalg.inv(self.K), pix_coords_h)

        points_c = (self.dot_const_c / np.dot(self.normal_c.T, pix_norm)) * pix_norm

        xy_world = np.dot(np.linalg.inv(self.R), points_c)[:2,:]
        return xy_world



if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node')
    lanenet_detector()
    rospy.spin()
