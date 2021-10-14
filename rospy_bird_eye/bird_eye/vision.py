import cv2
import numpy as np
import rospy
# from matplotlib.pyplot import imshow
# %matplotlib inline

def perspective_transform(img, shrink_pixel):
    height, width, channel = img.shape
    srcPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    dstPoint = np.array([[0, 0], [width, 0], [width-shrink_pixel, height], [0+shrink_pixel, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    img_per_transed = cv2.warpPerspective(img, matrix, (width, height))
    return img_per_transed

def image_merging(opt ,full_image, part_image, position):
    part_mask = np.all(part_image != 0, axis = -1).astype(bool)
    
    x_start = int((opt.width_bev-opt.width)/2)
    x_last = int(opt.width_bev-(opt.width_bev-opt.car_front_length)/2)
    y_start = int((opt.height_bev-opt.width)/2)
    if position == 'front':
        full_part_img = full_image[0:opt.height, x_start:x_start+opt.width,:] # (448, 800, 3) 
    elif position == 'back':
        full_part_img = full_image[opt.height_bev-opt.height:opt.height_bev, x_start:x_start+opt.width,:] # (448, 800, 3)
    elif position == 'right':
        full_part_img = full_image[y_start:y_start+opt.width, x_last:x_last+opt.height,:] # (800, 448, 3)
    elif position == 'left':
        full_part_img = full_image[y_start:y_start+opt.width,0:0+opt.height,:] # (448, 800, 3)
    else: raise ValueError('position should be {front},{back},{right},{left}')
    
    full_part_img[part_mask] = part_image[part_mask]

def color_masking(img,color):
    if color == 'yellow':
        # lower = np.array([28, 120, 120])
        # upper = np.array([32, 255, 255])
        lower = np.array([25, 120, 120])
        upper = np.array([35, 255, 255])
    elif color == 'green':
        lower = np.array([40, 40, 40])
        upper = np.array([70, 255, 255])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask[mask != 0] = 255
    return mask


def color_masking_image(img,color):
    if color == 'yellow':
        lower = np.array([25, 120, 120])
        upper = np.array([35, 255, 255])
        # lower = np.array([22, 93, 0])
        # upper = np.array([45, 255, 255])
    elif color == 'green':
        lower = np.array([40, 40, 40])
        upper = np.array([70, 255, 255])
    elif color == 'red':
        lower = np.array([161, 155, 84])
        upper = np.array([179, 255, 255])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    img_result = cv2.bitwise_and(img, img, mask=mask)
    return img_result

def canny_edge_detector(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny  

def draw_lines(image, lines, thickness): 
    line_image = np.zeros_like(image)
    color=[255, 0, 0]
    slopes = []
    if lines is not None: 
        for x1, y1, x2, y2 in lines:
                    slopes.append(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))
                    cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    for i in range(lines.shape[0]):
        print('line points: ', lines[i], '  slope: ', slopes[i])
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    return combined_image

def find_centroids(dst, gray):
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_EPS, 100, 0.005)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
    print('center:', centroids)
    return corners

# def detect_obstacle_pt(img, color):
#     '''
#     Find corners with HarrisCorner
#     '''
#     mask_img = color_masking_image(img, color)
#     # cv2.imshow('final', mask_img)
#     # cv2.waitKey(1)
#     # print('shape:', mask.shape)
#     gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)

#     dst = cv2.cornerHarris(gray, 10,3,0.04)
#     dst = cv2.dilate(dst, None)

#     corners = find_centroids(dst, gray)
#     for corner in corners:
#         img[int(corner[1]), int(corner[0])] = [0,0,255]
#     return corners, img, mask_img



def detect_obstacle_pt(img, color):
    '''
    Find corners with HarrisCorner
    '''
    mask_img = color_masking(img, color)
    # mask_img = mask_img.astype(bool)
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt_list = []
    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (0,0,255), 3)
        cnt_array = np.array(cnt).squeeze(1)
        cnt_list.append(cnt_array)
    try: 
	cnts = np.concatenate(cnt_list,0)
	min_value = np.min(cnts[:,-1])
    except: 
	min_value = None

    print('min_value:', min_value)
    return img, min_value
    # return None


def lines_finder(img, color, position):
    mask = color_masking(img, color)
    canny_edges = canny_edge_detector(mask)
    lines = cv2.HoughLinesP(
                    canny_edges,
                    rho=2,              #Distance resolution in pixels
                    theta=np.pi / 180,  #Angle resolution in radians
                    threshold=70,      #Min. number of intersecting points to detect a line  
                    lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2] 
                    minLineLength=60,   #Line segments shorter than this are rejected
                    maxLineGap=100       #Max gap allowed between points on the same line
                )
    
    lines = np.squeeze(lines)
    slopes = []
    line_length =[]
    try:
        for x1, y1, x2, y2 in lines:
                y_delta = y2 - y1
                x_delta = x2 - x1
                slopes.append(np.rad2deg(np.arctan2(y_delta, x_delta)))
                line_length.append(np.sqrt((x_delta**2 + y_delta**2)))
        for i in range(lines.shape[0]):
            #rospy.loginfo(position + ': points: {} | slope: {} | length: {}'.format(lines[i],slopes[i], line_length[i]))
	    pass
    except: 
        rospy.loginfo(position+': No Lines')
    return lines, slopes, line_length
