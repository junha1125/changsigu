#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
sys.path.append('/home/jetson/catkin_ws/src/rospy_bird_eye/bird_eye')
from vision import *
# from control import *
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Int16
# from rospy_bird_eye.msg import Encoder
# from rospy_bird_eye.msg import TotalControl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
# from statistics import mean
'''
aa = image_converter()
bb =aa.image_sub0
'''

class image_converter:
  def __init__(self):
    ## Vision team
    self.speedcmd_pub = rospy.Publisher("/speedcmd",Int16, queue_size=10)
    self.steercmd_pub = rospy.Publisher("/steercmd",Int16, queue_size=10)
    self.speedcmd = Int16
    self.steercmd = Int16

    self.speedcmd = 0
    self.steercmd = 90
    self.encoder = 0

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
    self.encoder_sub = rospy.Subscriber("/encoder",Float32, self.callback4)

    ## Control team
    # Simulation setting
    self.Start_x = 2
    self.Start_y = 0
    self.ref_speed = 55
    self.pline_count = 0
    self.detec_count = 0
    self.Obs = []
    self.Camera_Obs = []

    self.x = self.Start_x
    self.y = self.Start_y
    self.yaw = 90 * math.pi / 180

    self.ref_path_x,self.ref_yaw = self.GeneratePath()

    # Vehicle Parameter
    self.L = 0.65
    self.hurry = 0.6
    self.momtong = 0.25

    # Control Parameter
    self.KX = 0.5
    self.KE = 50
    self.di_max = 30
    self.di_min = -30

    # Simulation Parameter
    self.tolerance = 0.25

  def avg_slope(self,lines):
    try:
      return sum(lines) / len(lines)
    except:
      return None

    def slope_select(self, slope, length, direction):
        if direction=='right' or direction =='left':
            max_length_idx = np.argmax(length)
            slope_selected = slope[max_length_idx]
        else:
            return sum(lines) / len(lines)

        

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
    img_pered_left = perspective_transform(self.img2, opt.right_shrink_pixel)
    img_pered_right = perspective_transform(self.img3, opt.right_shrink_pixel)

    # Bird Eye view full Image
    bev_image = np.zeros((opt.height_bev,opt.width_bev,3))
    center_rectangle_point = np.array([[opt.height, opt.height], [opt.width_bev-opt.height, opt.height],\
        [opt.width_bev-opt.height, opt.height+opt.car_right_length], [opt.height, opt.height+opt.car_right_length]], dtype=np.int32)
    cv2.fillConvexPoly(bev_image, center_rectangle_point, (255,255,255))     
    car_ellipse_pint = cv2.ellipse2Poly((int(opt.width_bev/2),int(opt.height_bev/2)), (int(opt.car_front_length*0.7), int(opt.car_right_length*0.53)),0,0,360,30)
    cv2.fillConvexPoly(bev_image, car_ellipse_pint, (0,0,0))     

    # Make Bird Eye view
    image_merging(opt, bev_image, img_pered_front, 'front')
    image_merging(opt, bev_image, np.rot90(img_pered_back, k=2), 'back')
    image_merging(opt, bev_image, np.rot90(img_pered_right, k=3), 'right')
    image_merging(opt, bev_image, np.rot90(img_pered_left, k=1), 'left')
    self.bev_image_past_frame = bev_image
    rospy.loginfo(' --------  Save bev_image  ---------')
    cv2.imwrite('bev_image.jpg',bev_image)
    #bev_image = cv2.resize(bev_image, dsize=(int(opt.height_bev/6), int(opt.width_bev/6)), interpolation=cv2.INTER_NEAREST )
    #cv2.imshow('final', bev_image)
    #cv2.waitKey(1)
  

    # Fine lines
    lines_front, slopes_front, length_front = lines_finder(img_pered_front, 'green', 'front')
    lines_back, slopes_back, length_back = lines_finder(img_pered_back, 'green', 'back')
    lines_right, slopes_right, length_right = lines_finder(img_pered_right, 'green', 'right')
    lines_left, slopes_left, length_left = lines_finder(img_pered_left, 'green', 'left')

    slopes_front_avg = self.slope_select(slopes_front, length_front, direction='front')
    slopes_back_avg = self.slope_select(slopes_back, length_back, direction='back')
    slopes_right_avg = self.slope_select(slopes_right, length_right, direction='right')
    slopes_left_avg = self.slope_select(slopes_left, length_left, direction='left')


    return slopes_front_avg, slopes_back_avg, slopes_right_avg, slopes_left_avg

    pass

  def publish_drive(self):
    try:
      self.speedcmd_pub.publish(self.speedcmd)
      self.steercmd_pub.publish(self.steercmd)
    except CvBridgeError as e:
      print(e)

  def callback0(self,data):
    try:
      self.img0 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
      self.img0_get_bool = True
    except CvBridgeError as e:
      self.img0_get_bool = False
      print(e)

  def callback1(self,data):
    try:
      self.img1 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
      self.img1_get_bool = True
    except CvBridgeError as e:
      self.img1_get_bool = False
      print(e)

  def callback2(self,data):
    try:
      self.img2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
      self.img2_get_bool = True
    except CvBridgeError as e:
      self.img2_get_bool = False
      print(e)

  def callback3(self,data):
    try:
      self.img3 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
      self.img3_get_bool = True
      if all([self.img0_get_bool, self.img1_get_bool, self.img2_get_bool, self.img3_get_bool]):
        slopes_front_avg, slopes_back_avg, slopes_right_avg, slopes_left_avg = self.main_calculator()
        if slopes_left_avg is not None:
          slopes_left_avg = slopes_left_avg * (-1)
        print(slopes_right_avg)
        ds = self.encoder # encoder
        # Camera_Obs = obstacle detection
        if slopes_right_avg is not None:
          slopes_right_avg = slopes_right_avg * math.pi / 180
          print('slopes_right_avg = ', slopes_right_avg * 180 / math.pi)
        self.speedcmd,self.steercmd = self.PathTracking(ds, slopes_right_avg)
    
        # self.img0_get_bool, self.img1_get_bool, self.img2_get_bool, self.img3_get_bool = False, False, False, False

        # Publish Command
        self.publish_drive()

    except CvBridgeError as e:
      self.img3_get_bool = False
      print(e)
      
  def callback4(self,data):
   try:
     self.encoder = data.data
     #self.publish_drive()
     
   except CvBridgeError as e:
     print(e)

  def GeneratePath(self):
    ref_path_x = self.Start_x
    ref_yaw = 90 * math.pi / 180
    return ref_path_x, ref_yaw

  def localization(self,ds):
    dx = ds * math.cos(self.yaw)
    dy = ds * math.sin(self.yaw)
    print('yaw',self.yaw)
    print('dy',dy)
    self.x = self.update(self.x, dx)# byun soo i reum : encoder, vision_yaw, obstacle_detect
    self.y = self.update(self.y, dy)


  def Stop(self):
      v = 0
      return v

  def Constraint(self):
      if self.Obs[1] - self.y < self.tolerance:
        v = self.Stop()
      else:
        v = self.ref_speed
      return v

  def update(self,x,dx):
      x = x + dx
      return x

  def feedback_control(self,yaw_error,x_error):
      omega = self.KX * x_error + self.KE * yaw_error
      return omega

  def PathTracking(self,ds,vision_yaw):

      # Localization
      # fist vision yaw think.
      if vision_yaw is not None:
          self.pline_count = self.pline_count + 1
          self.yaw = self.ref_yaw - vision_yaw
          if self.pline_count == 1:
            print('oh oh line detection on on on ')
            self.x = self.Start_x + math.sqrt((self.x - self.Start_x) ** 2 + (self.y - self.Start_y) ** 2) * math.sin(vision_yaw)
            self.y = self.Start_y + math.sqrt((self.x - self.Start_x) ** 2 + (self.y - self.Start_y) ** 2) * math.cos(vision_yaw)
      yaw_error = self.ref_yaw - self.yaw

      # Camera Obstacle detection
      if self.detec_count == 0:  # initial obstacle position
          self.Obs = [self.Start_x + (0.979 / 2) - 0.272, self.Start_y + 2.53 + 1.28]

      if len(self.Camera_Obs) != 0:
          self.detec_count = detec_count + 1
          if self.Camera_Obs[0] < 0:
              Camera_x = self.Camera_Obs[0] - self.momtong
          else:
              Camera_x = self.Camera_Obs[0] + self.momtong

          Camera_y = self.Camera_Obs[1] + self.hurry    

          Obs_x = Camera_x * math.cos(yaw_error) + Camera_y * math.sin(yaw_error)
          Obs_y = Camera_x * math.sin(yaw_error) + Camera_y * math.cos(yaw_error)
          Obs = [self.x + Obs_x, self.y + Obs_y]

      self.localization(ds)
      x_error = self.x - self.ref_path_x

      ## Feedback Control
      di = self.feedback_control(yaw_error, x_error)
      di = di * (-1)
      
      if di >= self.di_max:
          di = self.di_max

      elif di <= self.di_min:
          di = self.di_min

      di = di + 90
      output_velocity = self.Constraint()
      output_steering = di
      
      
      print('encoder ds : ', self.encoder)
      print('x , y is :', self.x,'//',self.y)
      print('x_error : yaw_error',x_error,yaw_error)
      print('velocity & steering is : ', output_velocity,output_steering)
      return output_velocity,output_steering

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
