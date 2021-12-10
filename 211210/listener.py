#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import time

from numpy.core.einsumfunc import einsum_path
from numpy.core.fromnumeric import cumprod
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
import os
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
    self.vision_yaw_7 = 90
    self.right_length = 50
    self.left_length = 50

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
    self.idx_bird = 0
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
    self.direction = 2    # forward parking, direction = 2, backward parking

    # GoBack algorithm fla
    self.GoBack = 0
    self.GoBack2 = 0

    # Stop
    self.Stop = 0

    # Using Encoder date
    self.distance = 0
    self.distance2 = 0

    # Time variable
    self.time_start = 0
    self.time_end = 0

    ## Simulation setting
    self.ref_speed_F = 58
    self.ref_speed_B = -197
    # Phase 1
    self.ref_phase1_speed = self.ref_speed_F
    # Phase 2
    self.ref_phase2_speed = self.ref_speed_F
    # Phase 3
    self.ref_phase3_speed_F = self.ref_speed_F
    self.ref_phase3_speed_B = self.ref_speed_B
    # Phase 4
    self.ref_phase4_speed_F = self.ref_speed_F 
    self.ref_phase4_speed_B = self.ref_speed_B -5
    # Phase 5
    self.ref_phase5_speed_F = self.ref_speed_F
    self.ref_phase5_speed_B = self.ref_speed_B - 4
    # Phase 6
    self.ref_phase6_speed_F = self.ref_speed_F
    self.ref_phase6_speed_B = self.ref_speed_B + 8 
    # Phase 7
    self.ref_phase7_speed_F = self.ref_speed_F
    self.ref_phase7_speed_B = self.ref_speed_B + 5
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
    self.Phase5_B_KX = -self.KX
    self.Phase5_B_KE = -4
    # Phase 6
    self.Phase6_F_KX = self.KX
    self.Phase6_F_KE = 4
    self.Phase6_B_KX = -self.KX
    self.Phase6_B_KE = -4
    # Phase 7
    self.Phase7_F_KX = 0.12
    self.Phase7_F_KE = 1.2
    self.Phase7_B_KX = 0.12
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
    self.Phase5_tolerance1 = 0.46 # forward, 90cm
    self.Phase5_tolerance2 = 0.82 # backward, 40cm
    # Phase 6
    if self.direction == 1: 
      self.Phase6_tolerance1 = 6 # 진입할때의 tolerance : 진입한지 어느정도 지나고 yaw_error 가 클 때의 tolerance
    elif self.direction == 2:
      self.Phase6_tolerance1 = 3
    self.Phase6_tolerance2 = 6 # 진입할때의 tolerance : 진입한지 어느정도 지났을 때의 tolerance

    if self.direction == 1:
      self.Phase6_tolerance3 = 0.55 # 진입할때의 tolerance : 진입한지 어느정도 지나고 encoder distance

    elif self.direction == 2:
      self.Phase6_tolerance4 = 0.7
      
    # Phase 7
    self.Phase7_tolerance1 = 18 # 마지막 주차할때 Obstacle과의 거리(이거보다 작아야함)
    self.Phase7_tolerance2 = 405 # 마지막 주차할때 Obstacle과의 거리(이거보다 커야함)
    # Phase 10
    self.Phase10_tolerance1 = 3 # 뒤로 Back 했을 떄 어느정도 갈지에 대한 tolerance
    self.Phase10_tolerance2 = 0.25


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
        self.idx_bird += 1
        # self.bird_eye_view_drawer(self.idx_bird)
        back_yello_minvalue, front_yello_minvalue, front_red_point, back_red_point, back_info_tuple, front_info_tuple, left_info_tuple, right_info_tuple, left_global_tuple,right_global_tuple,g_right_red_point = self.line_obstacle_detector()
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
          if right_info_tuple[1] is not None:
            self.vision_yaw_7 = right_info_tuple[1]
        except:
          pass

        try:
          if right_info_tuple[0] is not None:
            self.right_length =  (right_info_tuple[0][0] + right_info_tuple[0][2])/2
        except:
          pass

        try:
          if left_info_tuple[0] is not None:
            self.left_length = 448 - (left_info_tuple[0][0] + left_info_tuple[0][2])/2
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
          var3 = back_red_point
          var4 = self.encoder
          var5 = back_info_tuple[1]
          var6 = None

        elif self.Phase10 == 1:
          var1 = front_info_tuple[1]
          var2 = front_red_point
          var3 = back_red_point
          var4 = self.encoder
          var5 = back_info_tuple[1]
          var6 = None

        elif self.Phase7 == 1:
          var1 = right_info_tuple[1]
          var2 = front_yello_minvalue
          if self.direction == 2:
            var2 = back_yello_minvalue

          if left_info_tuple[0] is not None :
            var3 = 448 - (left_info_tuple[0][0] + left_info_tuple[0][2])/2   
          else: 
            var3 = None
          if right_info_tuple[0] is not None :
            var4 = (right_info_tuple[0][0] + right_info_tuple[0][2])/2       
          else: 
            var4 = None
          var5 = front_red_point
          var6 = back_red_point

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
  def bird_eye_view_drawer(self,idx_bird):
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
    print('hhh')
    # cv2.imwrite('bev_image_{}.jpg'.format(now.strftime("%M_%S_%f")), bev_image[...,::-1])
    left_folder = '/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/image_folders/left_folder_auto_board_rere'
    front_folder = '/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/image_folders/front_folder_auto_board_rere'
    right_folder = '/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/image_folders/right_folder_auto_board_rere'
    back_folder = '/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/image_folders/back_folder_auto_board_rere'
     
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(front_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    os.makedirs(back_folder, exist_ok=True)
    
    # cv2.imwrite('/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/bev_folder_auto/bev_image_{}.jpg'.format(idx_bird), bev_image[...,::-1])
    # cv2.imwrite(os.path.join(front_folder, 'image_{}.jpg'.format(idx_bird)), self.img3[...,::-1])
    # cv2.imwrite(os.path.join(left_folder, 'image_{}.jpg'.format(idx_bird)), self.img0[...,::-1])
    # cv2.imwrite(os.path.join(right_folder, 'image_{}.jpg'.format(idx_bird)), self.img1[...,::-1])
    cv2.imwrite(os.path.join(back_folder, 'image_{}.jpg'.format(idx_bird)), self.img2[...,::-1])

    # cv2.imwrite('/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/back_folder_auto/back_image_{}.jpg'.format(idx_bird), self.img2[...,::-1])
    # cv2.imwrite('/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/right_folder_auto/right_image_{}.jpg'.format(idx_bird), self.img1[...,::-1])
    # cv2.imwrite('/home/future/catkin_ws/src/rospy_bird_eye/bird_eye/left_folder_auto/left_image_{}.jpg'.format(idx_bird), self.img0[...,::-1])

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
    back_point, back_slope, back_length = slope_select(lines_back, slopes_back, length_back, 95,       'back', 100)
    front_point, front_slope, front_length = slope_select(lines_front, slopes_front, length_front, 95, 'front', 100)
    left_point, left_slope, left_length = slope_select(lines_left, slopes_left, length_left, 90,       'left', 250)
    right_point, right_slope, right_length = slope_select(lines_right, slopes_right, length_right, 90, 'right', 250)
    left_G_point, left_G_slope, left_G_length = slope_select(lines_G_left, slopes_G_left, length_G_left, 90,       'G_left', 100)
    right_G_point, right_G_slope, right_G_length = slope_select(lines_G_right, slopes_G_right, length_G_right, 90, 'G_right', 100)
    # if left_G_slope is not None: rospy.loginfo('left_ global: {:0.4f} | {:0.4f} | {}'.format(left_G_slope, left_G_length, left_G_point))
    # if right_G_slope is not None: rospy.loginfo('right_global: {:0.4f} | {:0.4f} | {}'.format(right_G_slope, right_G_length, right_G_point))
    
    if back_point is not None:
      for line in [back_point]: 
        img_t = cv2.line(img_pered_back, tuple(line[:2]), tuple(line[2:]), (255,0,0),4)
      cv2.imwrite('img_pered_back.jpg',img_t)
    # for line in [right_point]: 
    #   img_t = cv2.line(img_pered_right, tuple(line[:2]), tuple(line[2:]), (255,0,0),4)
    #   cv2.imwrite('img_pered_right.jpg',img_t)
    #   import time; time.sleep(0.3)
  

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
    back_yello_minvalue, _ = detect_obstacle_pt(img_pered_back, 'yellow', 'back', True)
    if front_yello_minvalue is not None: front_yello_minvalue = 448 - front_yello_minvalue
    else: front_yello_minvalue = 1000
    
    if back_yello_minvalue is not None: back_yello_minvalue = 448 - back_yello_minvalue
    else: back_yello_minvalue = 0
    
    # Make need variable tuple
    back_info_tuple = (back_point, back_slope, back_length)
    front_info_tuple = (front_point, front_slope, front_length)
    left_info_tuple = (left_point, left_slope, self.left_length_m)
    right_info_tuple = (right_point, right_slope, right_length)
    left_global_tuple = (left_G_point, left_G_slope, left_G_length)
    right_global_tuple = (right_G_point, right_G_slope, right_G_length)
    g_right_red_point = (None, red_minvalue)

    return back_yello_minvalue, front_yello_minvalue, front_red_point, back_red_point, back_info_tuple, front_info_tuple, left_info_tuple, right_info_tuple, left_global_tuple, right_global_tuple, g_right_red_point
      



###########################################################################################
#                                       Control Team                                      #
###########################################################################################
  def feedback_control(self,yaw_error,x_error):
      omega = self.KX * x_error + self.KE * yaw_error
      return omega

  # global
  def preprocess_vision(self,vision_yaw):
    if vision_yaw is not None:
      if vision_yaw < 0:
        vision_yaw = vision_yaw + 180
    else:
      vision_yaw = self.vision_yaw

    return vision_yaw
  
  def preprocess_vision_7(self,vision_yaw):
    if vision_yaw is not None:
      if vision_yaw < 0:
        vision_yaw = vision_yaw + 180
    else:
      if self.vision_yaw_7 < 0:
        self.vision_yaw_7 = self.vision_yaw_7 + 180
      vision_yaw = self.vision_yaw_7

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
        di = 50
        output_steering = di

        self.distance = self.distance + var2 # x-direction distance

        if abs(self.distance) > self.Phase5_tolerance2: # 차가 앞으로 tolerance 이상 가면
          output_velocity = 0
          self.Phase5 = 0
          self.Phase6 = 1
          self.time_start = time.time()

    elif self.Phase6 == 1:
      ### Phase 6
      ## variable
      # var1 = vision_yaw(front)
      # var2 = red point of front Camera
      # var3 = red point of Back Camera
      # var4 = encoder distance
      # var5 = vision_yaw(back)
      # var6 = undetermined = None

      ### Validation
      print('======= Validation=======')
      print('Current Phase : Phase 6')
      print('vision_yaw(cameara) : ', var1)
      print('red point of front Camera : ', var2)
      print('red point of Back Camera : ', var3)
      print('Encoder distance : ', var4)
      print('vision_yaw(back) : ', var5)
      print('=========================')

      self.distance2 = self.distance2 + var4
      self.time_end = time.time()
      time_diff = self.time_end - self.time_start

      var1 = self.preprocess_vision(var1)
      var5 = self.preprocess_vision(var5)

      ref_yaw = 90

      if self.direction == 1:
        yaw_error = ref_yaw - var1
      elif self.direction == 2:
        yaw_error = ref_yaw - var5

      print(yaw_error)
      print('self direction : ', self.direction)

      if self.direction == 1: # 전방주차일 때,
        varvar = var2
      elif self.direction == 2:
        varvar = var3

      
      print(yaw_error)


      if time_diff > 1 and abs(yaw_error) < self.Phase6_tolerance1 and varvar is not None: # 조금만 있다가(to be robust), yaw error가 적어져서 드갈 수 있다고 판단되면,
        self.Phase6 = 0
        self.Phase7 = 1
        self.vision_yaw_7 = 90
        self.right_length = 50
        self.left_length = 50
        di = 93
        output_velocity = 0

     
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


        # Phase 6 -> 10 Constraint
        # if abs(yaw_error) > self.Phase6_tolerance1 + 1 and time_diff > self.Phase6_tolerance2: # 진입한지 5초가 지나고 yaw error가 크면
        if abs(yaw_error) > self.Phase6_tolerance1 + 1 and self.distance2 > self.Phase6_tolerance3:
          output_velocity = 0
          self.Phase6 = 0
          self.Phase10 = 1
          self.GoBack = 1
          self.distance2 = 0 
          self.time_start = time.time()

      elif self.direction == 2: # backward Parking
        output_velocity = self.ref_phase6_speed_B

        if self.GoBack == 0: # 한번도 GoBack Algorithm이 실행되지 않으면
          di = 130

        elif self.GoBack == 1: # 한번이라도 GoBack Algorithm이 실행됬으면 돌다가 레드 point를 보면 l(Back 자료를 안받아서 아직 미완성)
          if var3 is not None: # Parking area point가 보이면
            self.KX = self.Phase6_B_KX
            self.KE = self.Phase6_B_KE
            y_error = 0
            di = self.feedback_control(yaw_error,y_error) # 빨간색이 점 중앙으로 가게 Steering Control
            di = di
            if di >= self.di_max:
              di = self.di_max - 15
            elif di <= self.di_min:
              di = self.di_min + 15
            di = di + 90

          elif var3 is None: # 빨간색 점이 추출되지 않으면
            di = 110 # 덜 꺾기

        
        # Phase 6 -> 10 Consraint
        # if abs(yaw_error) > self.Phase6_tolerance1 and time_diff > self.Phase6_tolerance2: # 진입한지 5초가 지나고 yaw error가 크면
        if abs(yaw_error) > self.Phase6_tolerance1 + 1 and abs(self.distance2) > self.Phase6_tolerance4:
          output_velocity = 0
          self.Phase6 = 0
          self.Phase10 = 1
          self.GoBack = 1
          self.distance2 = 0
          self.time_start = time.time()

      output_steering = di

    elif self.Phase10 == 1: # Goback Phase
      # var1 = vision_yaw(front)
      # var2 = blue point of front camera
      # var3 = back red point
      # var4 = encoder distance
      # var5 = vision_yaw(back)
      # var6 = None


      ### Validation
      print('======= Validation=======')
      print('Current Phase : Phase 10')
      print('vision_yaw(front) : ', var1)
      print('red point of front Camera : ', var2)
      print('red point of Back Camera : ', var3)
      print('Encoder distance : ', var4)
      print(' vision_yaw(back) : ', var5)
      print('=========================')

      self.distance2 = self.distance2 + var4

      var1 = self.preprocess_vision(var1)
      var5 = self.preprocess_vision(var5)

      if var2 is None:
        var2 = [400]

      if var3 is None:
        var3 = [400]

      ref_yaw = 90

      if self.direction == 1:
        yaw_error = ref_yaw - var1

      elif self.direction ==2:
        yaw_error = ref_yaw - var5

      x_ref = 0
      current_x = 400 - var2[0]
      x_error = x_ref - current_x
  
      self.time_end = time.time()
      time_diff = self.time_end - self.time_start

      if self.direction == 1: # Forward Parking이면
        output_velocity = self.ref_phase10_speed_B # 뒤로 가야한다.
        di = 70
        if self.GoBack2 == 1 and abs(yaw_error) < self.Phase6_tolerance1:
          di = 93
          self.Phase7 = 1
          self.Phase10 = 0
          self.right_length = 50
          self.left_length = 50   
          self.vision_yaw_7 = 90     

      elif self.direction == 2: # Backward Parking 이면
        output_velocity = self.ref_phase10_speed_F # 앞으로 가야한다.
        di = 70
        if self.GoBack2 == 1 and abs(yaw_error) < self.Phase6_tolerance1:
          di = 93
          self.Phase7 = 1
          self.Phase10 = 0

      # if time_diff > self.Phase10_tolerance1 + 0.5 :  # 2초가 지나면 phase 6으로 return
      if abs(self.distance2) > self.Phase10_tolerance2:
        self.Phase6 = 1
        self.GoBack2 = 1
        self.Phase10 = 0
        self.Phase6_tolerance2 = 2  # tolerance를 짧게 변경하여 자주 보도록
        self.Phase6_tolerance3 = 0.1
        self.Phase6_tolerance4 = 0.1
        self.distance2 = 0
        self.time_start = time.time()

      output_steering = di 
    
    elif self.Phase7 == 1:
      # var1 = vision yaw(right surrounded view camera
      # var2 = Camera_y : distance btw vehicle & Obstacle (y-direction) of front/back camera
      # var3 = vision_distance_L : Distance btw vehicle and left parking line
      # var4 = vision_distance_R : Distance btw vehicle and right parking line
      # var5 = red point of front Camera
      # var6 = red point of back Camera

 
      print('vision_yaw(right)11 : ', var1)
      var1 = self.preprocess_vision_7(var1)
      var4, var3 = self.preprocess_length(var4, var3)

      print('======= Validation=======')
      print('Current Phase : Phase 7')
      print('vision_yaw(right) : ', var1)
      print('Obstacle distance : ', var2)
      print('left distance : ', var3)
      print('right distnace : ', var4)
      print(' redpoint x direction(front) : ', var5)
      print(' redpoint x diredtion(back)', var6)
      print('=========================')

      yaw_ref = 90
      x_ref = 0
      
      if self.direction == 1: # 전방 주차일 때,
        if var5 is not None:  # 앞 카메라로부터 Blue Point가 보이면
          current_x = var5[0] - 400
        elif var5 is None: # 검출이 안되면 30으로 가게한것 같은데, 검출이 되면 사이값 찾아가는 거고, 검출이 안되면 x_error가 0이다.
          current_x = var4 - var3

      elif self.direction == 2:
        if var6 is not None: # 뒷 카메라로부터 Blue Point가 보이면
          current_x = var6[0] - 400 
        elif var6 is None: # 검출이 안되면 30으로 가게한것 같은데, 검출이 되면 사이값 찾아가는 거고, 검출이 안되면 x_error가 0으로 가게 함.
          current_x = var4 - var3 

      x_error = current_x
      if self.direction == 1:
        yaw_error = yaw_ref - var1
      elif self.direction == 2:
        yaw_error = var1 - yaw_ref
      # print('**** x_error:: ', x_error)
      # print('**** yaw_error:: ', yaw_error)


      if self.direction == 2:
        self.Phase7_tolerance1 = self.Phase7_tolerance2

      ### Speed Equation
      if var2 is None : # Obstacle detection이 안되면
        if self.direction == 1: # 전방 주차일 때는
          output_velocity = self.ref_phase7_speed_F # 앞으로 가야한다.
        elif self.direction == 2: # 후방 주차일 때는
          output_velocity = self.ref_phase7_speed_B # 뒤로 가야한다.

      elif var2 is not None: # Obstacle detection이 되면
        if self.direction == 1 : # 전방 주차일때는
          if var2 > self.Phase7_tolerance1: # 어느정도 거리 이상이면
            output_velocity = self.ref_phase7_speed_F # 그냥 앞으로 간다.
          else: # 어느정도 가까워지면
            output_velocity = 0
            self.Stop = 1

        elif self.direction == 2: # 후방 주차일 때는
          if var2 < self.Phase7_tolerance2: # 어느정도 거리 이상이면(Back은 반대로)
            output_velocity = self.ref_phase7_speed_B # 그냥 뒤로 간다.
          else:
            output_velocity = 0
            self.Stop = 1

      ### Steering Equation
      if self.direction == 1:
        self.KX = self.Phase7_F_KX
        self.KE = self.Phase7_F_KE
      elif self.direction ==2:
        self.KX = self.Phase7_B_KX
        self.KE = self.Phase7_B_KE

      di = self.feedback_control(yaw_error,x_error)
      if di >= self.di_max - 15:
        di = self.di_max - 15
      elif di <= self.di_min + 15:
        di = self.di_min + 15
      di = di + 90
      output_steering = di

      if self.Stop == 1:
        output_velocity = 0
        output_steering = 93

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
