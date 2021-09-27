#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class image_converter:
  def __init__(self):
    self.steering_pub = rospy.Publisher("steering",Float32, queue_size=10)
    self.velocity_pub = rospy.Publisher("velocity",Float32, queue_size=10)
    self.steering = Float32(data=0)  #float() # Float32(data=0) 
    self.velocity = Float32(data=0) #float() # Float32(data=0) 

    self.bridge = CvBridge()
    self.img0 = 0
    self.img1 = 0
    self.img2 = 0
    self.img3 = 0
    self.image_sub0 = rospy.Subscriber("/camera1/usb_cam1/image_raw",Image,self.callback0)
    self.image_sub1 = rospy.Subscriber("/camera2/usb_cam2/image_raw",Image,self.callback1)
    self.image_sub2 = rospy.Subscriber("/camera3/usb_cam3/image_raw",Image,self.callback2)
    self.image_sub3 = rospy.Subscriber("/camera4/usb_cam4/image_raw",Image,self.callback3)

  def main_calculator(self):
    # self.img0, self.img1, self.img2, self.img3
    #print(self.img0.shape, self.img1.shape, self.img2.shape, self.img3.shape)
    img0_bev = self.bev_generation(self.img0)
    img1_bev = self.bev_generation(self.img1)
    img2_bev = self.bev_generation(self.img2)
    img3_bev = self.bev_generation(self.img3)
    # print(img0_bev.shape)
    bev_image = np.zeros((1696,1696,3))
    bev_image[0:448, 448:1248, :] = img0_bev # top
    bev_image[1248:1696, 448:1248, :] = np.flip(img1_bev,0) # down // np.flip(img, 0) 
    bev_image[448:1248, 1248:1696, :] = np.rot90(img2_bev,1) # right
    bev_image[448:1248, 0:448, :] = np.flip(np.rot90(img3_bev,1),0) # left
    cv2.imwrite('/home/jetson/Pictures/results/bev_img.jpeg', bev_image)

    #cv2.imshow('final', bev_image)
    #cv2.waitKey(1)
    pass

  def bev_generation(self, data):
    height, width = data.shape[0],data.shape[1]

    src = np.float32([[100,0], [700,0], [0,height], [width,height]])
    dst = np.float32([[0,0], [width,0], [0,height], [width,height]])
    M = cv2.getPerspectiveTransform(src, dst)

    data = data[0:height,0:width]
    warped_img = cv2.warpPerspective(data, M, (width, height))

    return warped_img

  def publish_drive(self):
    try:
      self.steering_pub.publish(self.steering)
      self.velocity_pub.publish(self.velocity)
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
      self.img3 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.main_calculator()
    except CvBridgeError as e:
      print(e)
  


def main(args):
  rospy.init_node('listener', anonymous=True)
  ic = image_converter()
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