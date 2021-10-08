#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Int16
from rospy_bird_eye.msg import Encoder
from rospy_bird_eye.msg import TotalControl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np



class image_converter:
  def __init__(self):
    # Publisher
    # CHECKIT!!: publisher 
    self.speedcmd_pub = rospy.Publisher("/speedcmd",Int16, queue_size=10)
    self.steercmd_pub = rospy.Publisher("/steercmd",Int16, queue_size=10)
    #self.vehiclecmd_pub = rospy.Publisher("/vehiclecmd",TotalControl, queue_size=10)

    self.speedcmd = Int16
    self.steercmd = Int16
    #self.vehiclecmd = TotalControl

    # Variables
    self.speedcmd = 90
    self.steercmd = 90
    #self.vehiclecmd.speedcmd = 90
    #self.vehiclecmd.steercmd = 90

    self.bridge = CvBridge()
    self.img0 = 0
    self.img1 = 0
    self.img2 = 0
    self.img3 = 0
    self.encoder = 0

    # Subscriber
    self.image_sub0 = rospy.Subscriber("/camera1/usb_cam1/image_raw",Image,self.callback0)
    self.image_sub1 = rospy.Subscriber("/camera2/usb_cam2/image_raw",Image,self.callback1)
    self.image_sub2 = rospy.Subscriber("/camera3/usb_cam3/image_raw",Image,self.callback2)
    self.encoder_sub = rospy.Subscriber("/encoder",Int16, self.callback3)
    self.image_sub3 = rospy.Subscriber("/camera4/usb_cam4/image_raw",Image,self.callback4)

  def main_calculator(self):

    # Image Synthesis and Calculation
    # self.img0, self.img1, self.img2, self.img3
    #print(self.img0.shape, self.img1.shape, self.img2.shape, self.img3.shape)
    img0_bev = self.bev_gen_lane_det(self.img0)
    img1_bev = self.bev_gen_lane_det(self.img1)
    img2_bev = self.bev_gen_lane_det(self.img2)
    img3_bev = self.bev_gen_lane_det(self.img3)
    # print(img0_bev.shape)
    bev_image = np.zeros((1696,1696,3))
    bev_image[0:448, 448:1248, :] = img0_bev # top
    bev_image[1248:1696, 448:1248, :] = np.flip(img1_bev,0) # down // np.flip(img, 0) 
    bev_image[448:1248, 1248:1696, :] = np.rot90(img2_bev,1) # right
    bev_image[448:1248, 0:448, :] = np.flip(np.rot90(img3_bev,1),0) # left
    cv2.imwrite('/home/jetson/Pictures/results/bev_img.jpeg', bev_image)

    #cv2.imshow('final', bev_image)
    #cv2.waitKey(1)

    # Pathplanning Algorithm


    # Result: vehicle cmd(steer, speed)
    #speedcmd = 90
    #steercmd = 90
    #self.publish_drive()
    
    pass

  def bev_gen_lane_det(self, data):
    '''
      1. BEV Generation for each imag
      2. Lane Detection
      3. Camera stitching
    '''
    height, width = data.shape[0],data.shape[1]

    src = np.float32([[100,0], [700,0], [0,height], [width,height]])
    dst = np.float32([[0,0], [width,0], [0,height], [width,height]])
    M = cv2.getPerspectiveTransform(src, dst)

    data = data[0:height,0:width]
    warped_img = cv2.warpPerspective(data, M, (width, height))

    return warped_img

  def bev_gen(self, data):
    height, width = data.shape[0],data.shape[1]

    src = np.float32([[100,0], [700,0], [0,height], [width,height]])
    dst = np.float32([[0,0], [width,0], [0,height], [width,height]])
    M = cv2.getPerspectiveTransform(src, dst)

    data = data[0:height,0:width]
    warped_img = cv2.warpPerspective(data, M, (width, height))

    return warped_img


  def publish_drive(self):
    try:
      self.speedcmd_pub.publish(self.speedcmd)
      self.steercmd_pub.publish(self.steercmd)
      #self.vehiclecmd_pub.publish(self.vehiclecmd)
    except CvBridgeError as e:
      print(e)

  def callback0(self,data):
    try:
      self.img0 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
    except CvBridgeError as e:
      print(e)

  def callback1(self,data):
    try:
      self.img1 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
    except CvBridgeError as e:
      print(e)

  def callback2(self,data):
    try:
      self.img2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
    except CvBridgeError as e:
      print(e)

  def callback3(self,data):
    try:
      self.encoder = data.data
      self.publish_drive()
      #print("pulsenum = ")
      #print(data.pulsenum)
    except CvBridgeError as e:
      print(e)

  def callback4(self,data):
    try:
      self.img3 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.main_calculator()

      # Pathplanning Algorithm
      # self.pathplanning

    except CvBridgeError as e:
      print(e)
  


def main(args):
  rospy.init_node('listener', anonymous=True)
  ic = image_converter() #Class definition
  
  #while not rospy.is_shutdown():
  #  ic.publish_drive()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

"""
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
"""