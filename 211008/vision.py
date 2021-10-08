import cv2
import numpy as np
# from matplotlib.pyplot import imshow
# %matplotlib inline

def perspective_transform(img, shrink_pixel):
    height, width, channel = img.shape
    srcPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    dstPoint = np.array([[0, 0], [width, 0], [width-shrink_pixel, height], [0+shrink_pixel, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    img_per_transed = cv2.warpPerspective(img, matrix, (width, height))
    return img_per_transed

def image_merging(opt ,full_image, part_image, position:str):
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

def color_masking(img,color:str):
    if color == 'yellow':
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
    elif color == 'green':
        lower = np.array([40, 100, 100])
        upper = np.array([70, 255, 255])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask[mask != 0] = 255
    return mask

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

def lines_finder(img, color:str):
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
    if lines is not None: 
        for x1, y1, x2, y2 in lines:
                    slopes.append(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))
    return lines, slopes