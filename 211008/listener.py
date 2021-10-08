#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
sys.path.append('/home/jetson/catkin_ws/src/rospy_bird_eye/bird_eye')
from vision import *
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
    self.bev_image_past_frame = 0
    self.img0 = 0; self.img0_get_bool = False
    self.img1 = 0; self.img1_get_bool = False
    self.img2 = 0; self.img2_get_bool = False
    self.img3 = 0; self.img3_get_bool = False
    self.image_sub0 = rospy.Subscriber("/camera1/usb_cam1/image_raw",Image,self.callback0)
    self.image_sub1 = rospy.Subscriber("/camera2/usb_cam2/image_raw",Image,self.callback1)
    self.image_sub2 = rospy.Subscriber("/camera3/usb_cam3/image_raw",Image,self.callback2)
    self.image_sub3 = rospy.Subscriber("/camera4/usb_cam4/image_raw",Image,self.callback3)

  def main_calculator(self):
    # Config
    class CONFIG():
      height, width = 448, 800
      front_shrink_pixel = 200
      right_shrink_pixel = 120
      car_front_length = width - 2 * front_shrink_pixel
      car_right_length = width - 2 * right_shrink_pixel
      height_bev = height * 2 + car_right_length
      width_bev = height * 2 + car_front_length
    opt = CONFIG

    # Perspective Transform Image
    img_pered_front = perspective_transform(self.img0, opt.front_shrink_pixel)
    img_pered_back = perspective_transform(self.img1, opt.front_shrink_pixel)
    img_pered_right = perspective_transform(self.img2, opt.right_shrink_pixel)
    img_pered_left = perspective_transform(self.img3, opt.right_shrink_pixel)

    # Bird Eye view full Image
    bev_image = np.zeros((opt.height_bev,opt.width_bev,3))
    center_rectangle_point = np.array([[opt.height, opt.height], [opt.width_bev-opt.height, opt.height],\
        [opt.width_bev-opt.height, opt.height+opt.car_right_length], [opt.height, opt.height+opt.car_right_length]], dtype=np.int32)
    cv2.fillConvexPoly(bev_image, center_rectangle_point, (255,255,255))     
    car_ellipse_pint = cv2.ellipse2Poly((int(opt.width_bev/2),int(opt.height_bev/2)), (int(opt.car_front_length*0.7), int(opt.car_right_length*0.53)),0,0,360,30)
    cv2.fillConvexPoly(bev_image, car_ellipse_pint, (0,0,0))     

    # Make Bird Eye view
    image_merging(opt, bev_image, img_pered_front, 'front')
    image_merging(opt, bev_image, np.rot90(img_pered_back, k=4), 'back')
    image_merging(opt, bev_image, np.rot90(img_pered_right, k=3), 'right')
    image_merging(opt, bev_image, np.rot90(img_pered_left, k=1), 'left')
    self.bev_image_past_frame = bev_image
    cv2.imshow('final', bev_image)
    cv2.waitKey(1)

    # Fine lines
    #  lines_front, slopes_front = lines_finder(img_pered_front, 'green')
    #  lines_back, slopes_back = lines_finder(img_pered_back, 'green')
    #  lines_right, slopes_right = lines_finder(img_pered_right, 'green')
    #  lines_left, slopes_left = lines_finder(img_pered_left, 'green')

    # 1. Averaging lines
    # 2. Making bird eye view
    # 3. return required_info
    pass

  def publish_drive(self):
    try:
      self.steering_pub.publish(self.steering)
      self.velocity_pub.publish(self.velocity)
    except CvBridgeError as e:
      print(e)

  def callback0(self,data):
    try:
      self.img0 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img0_get_bool = True
    except CvBridgeError as e:
      print(e)

  def callback1(self,data):
    try:
      self.img1 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img1_get_bool = True
    except CvBridgeError as e:
      print(e)

  def callback2(self,data):
    try:
      self.img2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img2_get_bool = True
    except CvBridgeError as e:
      print(e)

  def callback3(self,data):
    try:
      self.img3 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img3_get_bool = True
      if all([self.img0_get_bool, self.img1_get_bool, self.img2_get_bool, self.img3_get_bool]):
        self.main_calculator()
        self.img0_get_bool, self.img1_get_bool, self.img2_get_bool, self.img3_get_bool = False, False, False, False
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