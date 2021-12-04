#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import time

from numpy.core.einsumfunc import einsum_path
from numpy.lib.arraypad import _view_roi
from numpy.lib.type_check import nan_to_num
sys.path.append('/home/future/catkin_ws/src/rospy_bird_eye/bird_eye')
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
from datetime import datetime
import numpy as np
import math
import bisect
# from statistics import mean



###########################################################################################
#                                      Class Variables                                    #
###########################################################################################

class image_converter:
  def __init__(self):
    #### Vision team
    self.speedcmd_pub = rospy.Publisher("/speedcmd",Int16, queue_size=10)
    self.steercmd_pub = rospy.Publisher("/steercmd",Int16, queue_size=10)
    # Check It!
    self.flag1_pub = rospy.Publisher("/flag1",Int16, queue_size=10)
    self.flag2_pub = rospy.Publisher("/flag2",Int16, queue_size=10)
    self.speedcmd = Int16
    self.steercmd = Int16
    self.idx = 1

    self.speedcmd = 0
    self.steercmd = 93
    self.encoder = 0
    self.current_spd = 0
    self.Phase3_tempvar2 = [[250,0,250,0]]
    self.Phase3_tempvar3 = [[250,0,250,0]]
    self.vision_yaw = 90
    self.right_length = 250
    self.left_length = 250

    self.bridge = CvBridge()
    self.bev_image_past_frame = 0
    self.img0 = 0; self.img0_get_bool = False
    self.img1 = 0; self.img1_get_bool = False
    self.img2 = 0; self.img2_get_bool = False
    self.img3 = 0; self.img3_get_bool = False
    self.img4 = 0; self.img4_get_bool = False
    self.img5 = 0; self.img5_get_bool = False
    self.image_sub0 = rospy.Subscriber("/camera1/usb_cam1/image_raw",Image,self.callback0)
    self.image_sub1 = rospy.Subscriber("/camera2/usb_cam2/image_raw",Image,self.callback1)
    self.image_sub2 = rospy.Subscriber("/camera3/usb_cam3/image_raw",Image,self.callback2)
    self.image_sub3 = rospy.Subscriber("/camera4/usb_cam4/image_raw",Image,self.callback3)
    self.image_sub4 = rospy.Subscriber("/camera5/usb_cam5/image_raw",Image,self.callback4)
    self.image_sub5 = rospy.Subscriber("/camera6/usb_cam6/image_raw",Image,self.callback5)
    self.encoder_sub = rospy.Subscriber("/encoder",Float32, self.callback6)

    # momentum variable
    self.left_length_m = 800

    # Check It!
    # self.flag1_zero_sub = rospy.Subscriber("/flag1_zero",Image,self.callback7)
    # self.flag2_zero_sub = rospy.Subscriber("/flag2_zero",Image,self.callback8)


    # Config
    class CONFIG():
      height, width = 448, 800
      front_shrink_pixel = 200
      right_shrink_pixel = 120
      car_front_length = width - 2 * front_shrink_pixel
      car_right_length = width - 2 * right_shrink_pixel
      height_bev = height * 2 + car_right_length
      width_bev = height * 2 + car_front_length
    self.opt = CONFIG


    ### Control team
    # Phase state
    self.Phase1 = 1
    self.Phase2 = 0
    self.Phase3 = 0
    self.Phase4 = 0
    self.Phase5 = 0
    self.Phase6 = 0
    self.Phase7 = 0
    self.Phase8 = 0
    self.Phase10 = 0
    self.Phase11 = 0

    # Autonomous parking direction
    self.direction = 1  # forward parking, direction = 2, backward parking

    # GoBack algorithm flag
    self.GoBack = 0

    # Using Encoder date
    self.distance = 0

    # Time variable
    self.time_start = 0
    self.time_end = 0

    ## Simulation setting
    self.ref_speed_F = 58
    self.ref_speed_B = -197
    # Phase 1
    self.ref_phase1_speed = self.ref_speed_F
    # Phase 2
    self.ref_phase2_speed = self.ref_speed_F + 5
    # Phase 3
    self.ref_phase3_speed_F = self.ref_speed_F + 5
    self.ref_phase3_speed_B = self.ref_speed_B
    # Phase 4
    self.ref_phase4_speed_F = self.ref_speed_F
    self.ref_phase4_speed_B = self.ref_speed_B
    # Phase 5
    self.ref_phase5_speed_F = self.ref_speed_F
    self.ref_phase5_speed_B = self.ref_speed_B
    # Phase 6
    self.ref_phase6_speed_F = self.ref_speed_F
    self.ref_phase6_speed_B = self.ref_speed_B
    # Phase 7
    self.ref_phase7_speed_F = self.ref_speed_F
    self.ref_phase7_speed_B = self.ref_speed_B
    # Phase 10
    self.ref_phase10_speed_B = self.ref_speed_B
    self.ref_phase10_speed_F = self.ref_speed_F

    ##  Control Parameter
    self.KX = 0.5
    self.KE = 90
    self.di_max = 40
    self.di_min = -40
    # Phase 1 : No control for distance, Only for yaw Control
    self.Phase1_KX = 0
    self.Phase1_KE = 2
    # Phase 2 : No Control
    # Phase 3 : Control for distance and yaw
    self.Phase3_KX = 0.2
    self.Phase3_KE = 3
    # Phase 4 : Control for distance and yaw
    self.Phase4_KX = 0.2
    self.Phase4_KE = 4
    # Phase 5
    self.Phase5_F_KX = self.KX
    self.Phase5_F_KE = 4
    self.Phase5_B_KX = self.KX
    self.Phase5_B_KE = 4
    # Phase 6
    self.Phase6_F_KX = self.KX
    self.Phase6_F_KE = 4
    self.Phase6_B_KX = self.KX
    self.Phase6_B_KE = 4
    # Phase 7
    self.Phase7_F_KX = 1
    self.Phase7_F_KE = 4
    self.Phase7_B_KX = 0.5
    self.Phase7_B_KE = 4

    ## Simulation Parameter
    # Phase 1 : No tolerance
    # Phase 2
    self.Phase2_tolerance1 = 85  # phase2에서 3로 가기위한 constraint에 사용되는 yaw error tolerance
    # Phase 3
    self.Phase3_tolerance1 = 10 # phase3에서 4로 가기위한 Constraint에 사용 : yaw_error가 어느정도 적고 yellow point에서의 yaw error tolerance
    self.Phase3_tolerance2 = 200 # phase3에서 4로 가기위한 Constraint에 사용 : yaw_error가 어느정도 적고 yellow point에서의 yellow point distance
    # Phase 4
    self.Phase4_tolerance1 = 420 # Phase4에서 5로 가기 위한 Constraint에 사용 : redpoint가 어느정도 가까워질 때의 distance tolerance
    # PHase 5
    self.Phase5_tolerance1 = 0.6 # forward, 90cm
    self.Phase5_tolerance2 = 0.4 # backward, 40cm
    # Phase 6
    self.Phase6_tolerance1 = 14 # 진입할때의 tolerance : 진입한지 어느정도 지나고 yaw_error 가 클 때의 tolerance
    self.Phase6_tolerance2 = 10 # 진입할때의 tolerance : 진입한지 어느정도 지났을 때의 tolerance
    # Phase 7
    self.Phase7_tolerance1 = 17 # 마지막 주차할때 Obstacle과의 거리
    # Phase 10
    self.Phase10_tolerance1 = 3 # 뒤로 Back 했을 떄 어느정도 갈지에 대한 tolerance


  def callback0(self,data):
    try:
      self.img0 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")             # RGB 저장
      self.img0_get_bool = True
      # self.img0 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1] # BGR 저장
    except CvBridgeError as e:
      print(e)

  def callback1(self,data):
    try:
      self.img1 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img1_get_bool = True
      # self.img1 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
    except CvBridgeError as e:
      print(e)

  def callback2(self,data):
    try:
      self.img2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img2_get_bool = True
      # self.img2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
    except CvBridgeError as e:
      print(e)

  def callback3(self,data):
    try:
      self.img3 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img3_get_bool = True
      # self.img2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
    except CvBridgeError as e:
      print(e)

  def callback4(self,data):
    try:
      self.img4 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img4_get_bool = True
      # self.img2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")[...,::-1]
    except CvBridgeError as e:
      print(e)

  def callback5(self,data):
    try:
      self.img5 = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
      self.img5_get_bool = True

      if all([self.img0_get_bool,  self.img1_get_bool,  self.img2_get_bool,\
                self.img3_get_bool,  self.img4_get_bool,  self.img5_get_bool]) == True:
        # ==================== Implement : Start ====================
        # self.bird_eye_view_drawer()
        front_yello_minvalue, front_red_point, back_red_point, back_info_tuple, front_info_tuple, left_info_tuple, right_info_tuple, left_global_tuple,right_global_tuple,g_right_red_point = self.line_obstacle_detector()
        # print('back_red_point', back_red_point)
        # print("left_global_tuple  {:0.3f} {}".format(left_global_tuple[1], left_global_tuple[0]))
        # print("right_global_tuple {:0.3f} {}".format(right_global_tuple[1], right_global_tuple[0]))
        
        try:
          if left_global_tuple[0][0]  is not None or left_global_tuple[0][2] is not None:
            self.Phase3_tempvar2 = left_global_tuple
            
          if right_global_tuple[0][0] is not None or  right_global_tuple[0][2] is not None:
            self.Phase3_tempvar3 = right_global_tuple
        except:
          pass

        try:
          if right_global_tuple[1] is not None:
            self.vision_yaw = right_global_tuple[1]
        except:
          pass

        try:
          if right_info_tuple[2] is not None:
            self.right_length = right_info_tuple[2]
        except:
          pass

        try:
          if left_info_tuple[2] is not None:
            self.left_length = left_info_tuple[2]
        except:
          pass


        ## ==================== Implement : Control ====================
        ds = self.encoder

        ## Phase
        if self.Phase1 == 1:
          var1 = left_info_tuple[1]
          var2 = left_info_tuple[2]
          var3 = None
          var4 = None
          var5 = None
          var6 = None

        elif self.Phase2 == 1:
          var1 = right_global_tuple[1]
          var2 = None
          var3 = None
          var4 = None
          var5 = None
          var6 = None

        elif self.Phase3 == 1:
          var1 = right_global_tuple[1]
          var2 = 448 - (self.Phase3_tempvar2[0][0] + self.Phase3_tempvar2[0][2]) /2
          var3 = (self.Phase3_tempvar3[0][0] + self.Phase3_tempvar3[0][2]) /2
          var4 = g_right_red_point[1]
          var5 = front_yello_minvalue
          var6 = None

        elif self.Phase4 == 1:
          var1 = right_global_tuple[1]
          var2 = g_right_red_point[1]
          var3 = None
          var4 = None
          var5 = None
          var6 = None

        elif self.Phase5 == 1:
          var1 = right_global_tuple[1]
          var2 = self.encoder
          var3 = None
          var4 = None
          var5 = None
          var6 = None

        elif self.Phase6 == 1:
          var1 = front_info_tuple[1]
          var2 = front_red_point 
          var3 = None # back_red_tuple 만들어야됨
          var4 = None
          var5 = None
          var6 = None

        elif self.Phase10 == 1:
          var1 = None
          var2 = None
          var3 = None
          var4 = None
          var5 = None
          var6 = None

        elif self.Phase7 == 1:
          var1 = right_info_tuple[1]
          var2 = front_yello_minvalue
          var3 = left_info_tuple[2]
          var4 = right_info_tuple[2]
          var5 = front_red_point
          var6 = None

        ## Speed cmd steer cmd
        self.speedcmd, self.steercmd = self.AutonomousParking(var1, var2, var3, var4, var5, var6)
        if self.steercmd > 0 :
          self.steercmd = int(math.floor(self.steercmd))
        else:
          self.steercmd = int(math.ceil(self.steercmd))
        # import pdb
        # pdb.set_trace()
        self.publish_drive()
        # if slopes_left_avg is not None:
        #   slopes_left_avg = slopes_left_avg * (-1)
        # print(slopes_right_avg)
        # self.Camera_Obs = min_value
        # # Camera_Obs = obstacle detection
        # if slopes_right_avg is not None:
        #   slopes_right_avg = slopes_right_avg * math.pi / 180
        #   print('slopes_right_avg = ', slopes_right_avg * 180 / math.pi)
        # self.speedcmd,self.steercmd = self.PathTracking(ds, slopes_right_avg)
        # ==================== Implement End ====================
        self.img0_get_bool, self.img1_get_bool, self.img2_get_bool, self.img3_get_bool, self.img4_get_bool,  self.img5_get_bool = False, False, False, False, False, False
        self.publish_drive()

    except CvBridgeError as e:
      print(e)

  def callback6(self,data):
    try:
      self.encoder = data.data
    except CvBridgeError as e:
     print(e)



  def publish_drive(self):
    try:
      self.speedcmd_pub.publish(self.speedcmd)
      self.steercmd_pub.publish(self.steercmd)
    except CvBridgeError as e:
      print(e)


###########################################################################################
#                                       Vision Team                                       #
###########################################################################################
  def bird_eye_view_drawer(self):
    # Perspective Transform Image
    img_pered_left = perspective_transform(self.img0, self.opt.right_shrink_pixel)
    img_pered_right = perspective_transform(self.img1, self.opt.right_shrink_pixel)
    img_pered_back = perspective_transform(self.img2, self.opt.front_shrink_pixel)
    img_pered_front = perspective_transform(self.img3, self.opt.front_shrink_pixel)

    # Bird Eye view full Image
    bev_image = np.zeros((self.opt.height_bev,self.opt.width_bev,3))
    center_rectangle_point = np.array([[self.opt.height, self.opt.height], [self.opt.width_bev-self.opt.height, self.opt.height],\
        [self.opt.width_bev-self.opt.height, self.opt.height+self.opt.car_right_length], [self.opt.height, self.opt.height+self.opt.car_right_length]], dtype=np.int32)
    cv2.fillConvexPoly(bev_image, center_rectangle_point, (255,255,255))     
    car_ellipse_pint = cv2.ellipse2Poly((int(self.opt.width_bev/2),int(self.opt.height_bev/2)), (int(self.opt.car_front_length*0.7), int(self.opt.car_right_length*0.53)),0,0,360,30)
    cv2.fillConvexPoly(bev_image, car_ellipse_pint, (0,0,0))     

    # Make Bird Eye view
    image_merging(self.opt, bev_image, img_pered_front, 'front')
    image_merging(self.opt, bev_image, np.rot90(img_pered_back, k=2), 'back')
    image_merging(self.opt, bev_image, np.rot90(img_pered_right, k=3), 'right')
    image_merging(self.opt, bev_image, np.rot90(img_pered_left, k=1), 'left')
    self.bev_image_past_frame = bev_image
    now = datetime.now()
    cv2.imwrite('bev_image_{}.jpg'.format(now.strftime("%M_%S_%f")), bev_image[...,::-1])
    # bev_image = cv2.resize(bev_image[...,::-1], dsize=(int(self.opt.height_bev/6), int(self.opt.width_bev/6)), interpolation=cv2.INTER_NEAREST )
    # cv2.imshow('final', bev_image)
    # cv2.waitKey(1)
    rospy.loginfo(' --------  Saved bev_image  ---------===============================')
    return 


  def line_obstacle_detector(self):
    # Perspective Transform Image
    img_pered_front =  cv2.cvtColor(perspective_transform(self.img3, self.opt.front_shrink_pixel), cv2.COLOR_BGR2RGB)
    img_pered_back =   cv2.cvtColor(np.rot90(perspective_transform(self.img2, self.opt.front_shrink_pixel), k=2), cv2.COLOR_BGR2RGB)
    img_pered_right =  cv2.cvtColor(np.rot90(self.img1, k=3), cv2.COLOR_BGR2RGB)
    img_pered_left =   cv2.cvtColor(np.rot90(self.img0, k=1), cv2.COLOR_BGR2RGB)
    img_global_right = cv2.cvtColor(np.rot90(self.img4, k=3), cv2.COLOR_BGR2RGB)
    img_global_left =  cv2.cvtColor(np.rot90(self.img5, k=1), cv2.COLOR_BGR2RGB)
    
    # Find lines
    lines_front, slopes_front, length_front = lines_finder(img_pered_front, 'green', 'front')
    lines_back, slopes_back, length_back = lines_finder(img_pered_back, 'green', 'back')
    lines_right, slopes_right, length_right = lines_finder(img_pered_right, 'green', 'right')
    lines_left, slopes_left, length_left = lines_finder(img_pered_left, 'green', 'left')
    lines_G_right, slopes_G_right, length_G_right = lines_finder(img_global_right, 'green', 'G_right')
    lines_G_left, slopes_G_left, length_G_left = lines_finder(img_global_left, 'green', 'G_left')

    # for line in lines_front: 
    #   img_t = cv2.line(img_pered_front, tuple(line[:2]), tuple(line[2:]), (255,0,0),4)
    # cv2.imwrite('front_img_wi_line.jpg',img_t)

    # ============================================ Refer below (Implemntt) ===========================
    # Select lines
    back_point, back_slope, back_length = slope_select(lines_back, slopes_back, length_back, 95, 'back', 100)
    front_point, front_slope, front_length = slope_select(lines_front, slopes_front, length_front, 95, 'front', 100)
    left_point, left_slope, left_length = slope_select(lines_left, slopes_left, length_left, 90, 'left', 250)
    right_point, right_slope, right_length = slope_select(lines_right, slopes_right, length_right, 90, 'right', 100)
    left_G_point, left_G_slope, left_G_length = slope_select(lines_G_left, slopes_G_left, length_G_left, 90, 'G_left', 100)
    right_G_point, right_G_slope, right_G_length = slope_select(lines_G_right, slopes_G_right, length_G_right, 90, 'G_right', 100)
    # if left_G_slope is not None: rospy.loginfo('left_ global: {:0.4f} | {:0.4f} | {}'.format(left_G_slope, left_G_length, left_G_point))
    # if right_G_slope is not None: rospy.loginfo('right_global: {:0.4f} | {:0.4f} | {}'.format(right_G_slope, right_G_length, right_G_point))
  

    # length momentum update
    m = 0.95
    if left_length is not None and self.left_length_m is not None : self.left_length_m = self.left_length_m*m + left_length*(1-m)
    if self.left_length_m == None or self.left_length_m < 580  : self.left_length_m = None

    # Find obstacle point
    parking_area_color = 'blue'
    _, front_red_point = detect_obstacle_pt(img_pered_front, parking_area_color, 'front', True)
    _, back_red_point = detect_obstacle_pt(img_pered_back,   parking_area_color, 'back', True)
    red_minvalue, _ = detect_obstacle_pt(img_global_right,   parking_area_color, 'right', True)
    front_yello_minvalue, _ = detect_obstacle_pt(img_pered_front, 'yellow', 'front', True)
    if front_yello_minvalue is not None: front_yello_minvalue = 448 - front_yello_minvalue
    else: front_yello_minvalue = 1000
    
    
    # Make need variable tuple
    back_info_tuple = (back_point, back_slope, back_length)
    front_info_tuple = (front_point, front_slope, front_length)
    left_info_tuple = (left_point, left_slope, self.left_length_m)
    right_info_tuple = (right_point, right_slope, right_length)
    left_global_tuple = (left_G_point, left_G_slope, left_G_length)
    right_global_tuple = (right_G_point, right_G_slope, right_G_length)
    g_right_red_point = (None, red_minvalue)

    return front_yello_minvalue, front_red_point, back_red_point, back_info_tuple, front_info_tuple, left_info_tuple, right_info_tuple, left_global_tuple, right_global_tuple, g_right_red_point
      



###########################################################################################
#                                       Control Team                                      #
###########################################################################################
  def feedback_control(self,yaw_error,x_error):
      omega = self.KX * x_error + self.KE * yaw_error
      return omega

  def preprocess_vision(self,vision_yaw):
    if vision_yaw is not None:
      if vision_yaw < 0:
        vision_yaw = vision_yaw + 180
    else:
      vision_yaw = self.vision_yaw

    return vision_yaw

  def preprocess_length(self,length_R,length_L):
    if length_R is None:
      length_R = self.right_length

    if length_L is None:
      length_L = self.left_length

    return length_R, length_L


  def AutonomousParking(self, var1, var2, var3, var4, var5, var6):
    if self.Phase1 == 1:
      ### Phase 1
      ## variable
      # var1 = vision_yaw
      # var2 = left line length
      # var3 = undetermined = None
      # var4 = undetermined = None
      # var5 = undetermined = None
      # var6 = undetermined = None

      print('======= Validation=======')
      print('Current Phase : Phase 1')
      print('vision_yaw(right,global) : ', var1)
      print('left line length : ', var2)
      print('=========================')

      ref_yaw = 90
      output_velocity = self.ref_phase1_speed

      var1 = self.preprocess_vision(var1) # preprocess
      x_error = 0
      yaw_error = ref_yaw - var1
      self.KX = self.Phase1_KX
      self.KE = self.Phase1_KE
      di = self.feedback_control(yaw_error, x_error)
      if di >= self.di_max:
        di = self.di_max
      elif di <= self.di_min:
        di = self.di_min

      di = di + 90

      # Phase 1 end Constraint
      if var2 is None:
        self.Phase1 = 0
        self.Phase2 = 1
        output_velocity = 0
        di = 93

      output_steering = di

    elif self.Phase2 == 1:
      ### Phase 2
      ## variable
      # var1 = vision_yaw
      # var2 = undetermined = None
      # var3 = undetermined = None
      # var4 = undetermined = None
      # var5 = undetermined = None
      # var6 = undetermined = None

      print('======= Validation=======')
      print('Current Phase : Phase 2')
      print('vision_yaw(right,global) : ', var1)
      print('=========================')

      output_velocity = self.ref_phase2_speed
      ref_yaw = 90

      # Phase 2 end Constraint
      if var1 is None: # 안보이면 그냥 왼쪽으로 쭉 꺾음
        di = 50
      else:
        yaw_error = abs(ref_yaw - var1)
        if var1 > self.Phase2_tolerance1:
          self.Phase2 = 0
          self.Phase3 = 1
          di = 80

        else:
          di = 50

      output_steering = di

    elif self.Phase3 == 1:
    ### Phase 3
    ## variable
    # var1 = vision_yaw(right camera)
    # var2 = vision_distance_L : Distance btw vehicle and left parking line
    # var3 = vision_distance_R : Distance btw vehicle and right parking line
    # var4 = parking_point : redpoint distance
    # var5 = yellow point
    # var6 = None

      print('======= Validation=======')
      print('Current Phase : Phase 3')
      print('vision_yaw(right,global) : ', var1)
      print('left distance : ', var2)
      print('right distance : ', var3)
      print('red point distance : ', var4)
      print('yellow point distance : ', var5)
      print('=========================')

      ref_yaw = 90 # or 180 degrees
      ref_y = 2

      var1 = self.preprocess_vision(var1)
      output_velocity = self.ref_phase3_speed_F
      current_y = var2 - var3
      y_error = ref_y - current_y
      yaw_error = ref_yaw - var1
      self.KX = self.Phase3_KX
      self.KE = self.Phase3_KE
      di = self.feedback_control(yaw_error, y_error)
      if di >= self.di_max:
        di = self.di_max - 15
      elif di <= self.di_min:
        di = self.di_min + 15
      di = di + 90

      # Phase 3 end constraint
      if yaw_error < self.Phase3_tolerance1: # yaw error가 어느정도 적을 때,
        if var5 < self.Phase3_tolerance2: # 이 때 yellow point가 가까우면
          output_velocity = 0
          self.Phase3 = 0
          self.Phase4 = 1

      output_steering = di

    elif self.Phase4 == 1:
    ### Phase 4
    ## variable
    # var1 = vision_yaw(right camera)
    # var2 = parking_point : redpoint distance
    # var3 is not determined
    # var4 is not determined
    # var5 is not determined
    # var6 is not determined

    ## validation
      print('======= Validation=======')
      print('Current Phase : Phase 4')
      print('vision_yaw(right,global) : ', var1)
      print('red point distance : ', var2)
      print('=========================')

      ref_yaw = 90
      ref_y = 2

      output_velocity = self.ref_phase4_speed_B
      var1 = self.preprocess_vision(var1)
      y_error = 0 # Yaw만 바로 잡기 위해
      yaw_error = - ref_yaw + var1
      self.KX = self.Phase4_KX
      self.KE = self.Phase4_KE
      di = self.feedback_control(yaw_error, y_error)
      if di >= self.di_max:
        di = self.di_max - 15
      elif di <= self.di_min:
        di = self.di_min + 15
      di = di + 90
      output_steering = di
      print('Phase4 output-steering',output_steering)

      # Phase4 end constraint
      if var2 is not None: # red point가 보이고
        if var2 < self.Phase4_tolerance1: # 거리가 어느정도 가까워지면
          output_velocity = 0
          self.Phase4 = 0
          self.Phase5 = 1

    elif self.Phase5 == 1:
      ### Phase 5
      ## variable
      # var1 = vision_yaw
      # var2 = encoder ds
      # var3 = undetermined = None
      # var4 = undetermined = None
      # var5 = undetermined = None
      # var6 = undetermined = None

      ### Validation
      print('======= Validation=======')
      print('Current Phase : Phase 5')
      print('vision_yaw : ', var1)
      print('encoder ds: ', var2)
      print("self distance: ", self.distance)
      print('=========================')

      var1 = self.preprocess_vision(var1)

      ref_yaw = 90
      yaw_error = var1 - ref_yaw 
      y_error = 0 # Yaw error만 바로 잡겠다.

      if self.direction == 1: # forward Parking
        output_velocity = self.ref_phase5_speed_B  # 뒤로 가야함

        self.KX = self.Phase5_F_KX
        self.KE = self.Phase5_F_KE
        di = self.feedback_control(yaw_error, y_error)
        di = di + 90
        output_steering = di

        self.distance = self.distance + var2 * abs(math.sin(var1 * math.pi * 2 / 360)) # x-direction distance

        if abs(self.distance) > self.Phase5_tolerance1: # 차가 뒤로 100cm 이상 가면
          output_velocity = 0
          self.Phase5 = 0
          self.Phase6 = 1
          self.time_start = time.time()

      elif self.direction == 2: # backward Parking
        output_velocity = self.ref_phase5_speed_F  # 앞으로 가야함

        self.KX = self.Phase5_B_KX
        self.KE = self.Phase5_F_KE
        di = self.feedback_control(yaw_error, y_error)
        di = di + 90
        output_steering = di

        self.distance = self.distance + var2 * math.sin(var1 * math.pi * 2 / 360) # x-direction distance

        if abs(self.distance) > self.Phase5_tolerance2: # 차가 앞으로 40cm 이상 가면
          output_velocity = 0
          self.Phase5 = 0
          self.Phase6 = 1
          self.time_start = time.time()

    elif self.Phase6 == 1:
      ### Phase 6
      ## variable
      # var1 = vision_yaw
      # var2 = red point of front Camera
      # var3 = red point of Back Camera
      # var4 = undetermined = None
      # var5 = undetermined = None
      # var6 = undetermined = None

      ### Validation
      print('======= Validation=======')
      print('Current Phase : Phase 6')
      print('vision_yaw : ', var1)
      print('red point of front Camera : ', var2)
      print('red point of Back Camera : ', var3)
      print('=========================')

      self.time_end = time.time()
      time_diff = self.time_end - self.time_start

      var1 = self.preprocess_vision(var1)

      ref_yaw = 90
      yaw_error = ref_yaw - var1

      print('time diff : ',time_diff)
      print('yaw_error : ', yaw_error)

      if time_diff > self.Phase6_tolerance2 - 1 and abs(yaw_error) < self.Phase6_tolerance1:
        self.Phase6 = 0
        self.Phase7 = 1
        di = 93
        output_velocity = 0
        print('ohoh why not pass')


      elif self.direction == 1: # forward Parking
        output_velocity = self.ref_phase6_speed_F # 앞으로 가야됨.

        if self.GoBack == 0: # 한번도 GoBack Algorithm이 실행되지 않으면
          di = 130
        elif self.GoBack == 1: # 한번이라도 GoBack Algorithm이 실행됬으면
          if var2 is not None: # 빨GAN색 점이 보이면
            self.KX = self.Phase6_F_KX
            self.KE = self.Phase6_F_KE
            # y_error = var2[0] - 400
            y_error = 0
            di = self.feedback_control(yaw_error, y_error) # 빨간색이 점 중앙으로 가게 Steering Control
            if di >= self.di_max:
              di = self.di_max - 15
            elif di <= self.di_min:
              di = self.di_min + 15
            di = di + 90

          elif var2 is None: # 빨간색 점이 추출되지 않으면
            di = 110 # 덜 꺾기

            # phase 6 -> 7 Constraint # 마지막 단계 진입 constraint
    
        # Phase 6 -> 10 Constraint
        if abs(yaw_error) > self.Phase6_tolerance1 + 1 and time_diff > self.Phase6_tolerance2: # 진입한지 5초가 지나고 yaw error가 크면
          output_velocity = 0
          self.Phase6 = 0
          self.Phase10 = 1
          self.GoBack = 1
          self.time_start = time.time()

      elif self.direction == 2: # backward Parking
        output_velocity = self.ref_phase6_speed_B # 뒤로 가야됨.

        if self.GoBack == 0: # 한번도 GoBack Algorithm이 실행되지 않으면
          di = 50
        elif self.GoBack == 1: # 한번이라도 GoBack Algorithm이 실행됬으면 돌다가 레드 point를 보면 l(Back 자료를 안받아서 아직 미완성)
          if var3[0] is not None: # 빨간색 점이 보이면
            self.KX = self.Phase6_B_KX
            self.KE = self.Phase6_B_KE
            y_error = 400 - var3[0]
            di = self.feedback_control(yaw_error,y_error) # 빨간색이 점 중앙으로 가게 Steering Contro
            if di >= self.di_max:
              di = self.di_max - 15
            elif di <= self.di_min:
              di = self.di_min + 15
            di = di + 90

          elif var3[0] is None: # 빨간색 점이 추출되지 않으면
            di = 70 # 덜 꺾기

        # Phase 6 -> 10 Consraint
        if abs(yaw_error) > self.Phase6_tolerance1 and time_diff > self.Phase6_tolerance2: # 진입한지 5초가 지나고 yaw error가 크면
          output_velocity = 0
          self.Phase6 = 0
          self.Phase10 = 1
          self.GoBack = 1
          self.time_start = time.time()

      output_steering = di

    elif self.Phase10 == 1: # Goback Phase
      # var1 = undetermined = None
      # var2 = undetermined = None
      # var3 = undetermined = None
      # var4 = undetermined = None
      # var5 = undetermined = None
      # var6 = undetermined = None

      ### Validation
      print('======= Validation=======')
      print('Current Phase : Phase 10')
      print('No Sensor value used')
      print('=========================')

      self.time_end = time.time()
      time_diff = self.time_end - self.time_start
      

      if self.direction == 1: # Forward Parking이면
        output_velocity = self.ref_phase10_speed_B # 뒤로 가야한다.
        di = 70

      elif self.direction == 2: # Backward Parking 이면
        output_velocity = self.ref_phase10_speed_F # 앞으로 가야한다.
        di = 110

      if time_diff > self.Phase10_tolerance1 + 0.5 :  # 2초가 지나면 phase 6으로 return
        self.Phase6 = 1
        self.Phase10 = 0
        self.Phase6_tolerance2 = 2  # tolerance를 짧게 변경하여 자주 보도록
        self.time_start = time.time()

      output_steering = di 
    
    elif self.Phase7 == 1:
      # var1 = vision yaw
      # var2 = Camera_y : distance btw vehicle & Obstacle (y-direction)
      # var3 = vision_distance_L : Distance btw vehicle and left parking line
      # var4 = vision_distance_R : Distance btw vehicle and right parking line
      # var5 = red point of front Camera
      # var6 = undetermined = None

      print('======= Validation=======')
      print('Current Phase : Phase 7')
      print('vision_yaw : ', var1)
      print('Obstacle distance : ', var2)
      print('left distance : ', var3)
      print('right distnace : ', var4)
      print('redpoint x direction : ', var5)
      print('=========================')

      var1 = self.preprocess_vision(var1)
      var4, var3 = self.preprocess_length(var4, var3)

      if var5 is None:
        var5 = [400]


      yaw_ref = 90
      x_ref = 0
      current_x = 400 - var5[0]
      x_error = x_ref - current_x
      yaw_error = yaw_ref - var1

      if var2 is None : # Obstacle detection이 안되면
        if self.direction == 1: # 전방 주차일 때는
          output_velocity = self.ref_phase7_speed_F # 앞으로 가야한다.
        elif self.direction ==2: # 후방 주차일 때는
          output_velocity = self.ref_phase7_speed_B # 뒤로 가야한다.

      elif var2 is not None: # Obstacle detection이 되면
        if var2 > self.Phase7_tolerance1: # 어느정도 거리 이상이면
          if self.direction == 1: # 전방 주차일 때는
            output_velocity = self.ref_phase7_speed_F # 그냥 앞으로 간다.
          elif self.direction ==2: # 후방 주차일 때는
            output_velocity = self.ref_phase7_speed_B # 그냥 뒤로 간다.
        else: # 어느정도 가까워지면
          output_velocity = 0

      if self.direction == 1:
        self.KX = self.Phase7_F_KX
        self.KE = self.Phase7_F_KE
      elif self.direction ==2:
        self.KX = self.Phase7_B_KX
        self.KE = self.Phase7_B_KE

      di = self.feedback_control(yaw_error,x_error)
      if di >= self.di_max:
        di = self.di_max
      elif di <= self.di_min:
        di = self.di_min
      di = di + 90
      output_steering = di

    return output_velocity, output_steering

###########################################################################################
#                                          MAIN                                           #
###########################################################################################
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
