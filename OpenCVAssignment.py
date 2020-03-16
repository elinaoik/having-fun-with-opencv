#########################################################
# Author: Elina Oikonomou, r0737756                     #
# Last Modified: 10th March 2020                        #
# Course: Computer Vision                               #
# MSc of Artificial Intelligence, KU LEUVEN             #
# Individual Assignment 1                               #
#########################################################

# Import Libraries
import numpy as np
import cv2 as cv
import random as rng

# Read video
cap = cv.VideoCapture('finalmovie.mp4')

# Find width, height and Frames-per-second of the video
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv.CAP_PROP_FPS))

# Create a Video writer
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('movie1.mp4', fourcc, FPS, (width, height))


# ---------------------- DEFINE FUNCTIONS ---------------------------- #

# ------------------ PART 1: Basic Image Processing ------------------ #

# Function that converts a frame to grayscale
def GrayScale(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_ = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    
    return frame_

# Function that computes a Gaussian Blur with Kernel of ksize
def Gaussian(img, ksize=15):
    frame_ = cv.GaussianBlur(img, (ksize,ksize), 0)
    
    return frame_

# Function that computes a Bilateral Blur with Kernel of ksize
def Bilateral(img, ksize=15):
    frame_ = cv.bilateralFilter(img, ksize, 75, 75)
    
    return frame_

# Function that grabs an object in RBG color space
def RBGthresholding(img):
    #Define the approximate range of RBG values of the cup
    bgr = [20, 40, 180]
    threshold_ = 40  #The optimal threshold has the value of the blue channe
 
    #Compute min, max values
    min_bgr = np.array([bgr[0] - threshold_, bgr[1] - threshold_, bgr[2] - threshold_])
    max_bgr = np.array([bgr[0] + threshold_, bgr[1] + threshold_, bgr[2] + threshold_])
 
    mask = cv.inRange(img, min_bgr, max_bgr) # Find the mask 
    frame_ = cv.bitwise_and(img, img, mask = mask) #Apply the mask to get the cup from 
    
    return frame_

# Function that grabs an object in HSV color space
def HSVthresholding(img):    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Red Filter
    lower_red = np.array([0, 120, 150])
    upper_red = np.array([9, 255, 255])
    
    mask = cv.inRange(hsv, lower_red, upper_red)
    frame_ = cv.bitwise_and(frame, frame, mask = mask)
    
    return frame_

''' How to find HSV colors
red = np.uint8([[[33,62,212 ]]])
hsv_red = cv.cvtColor(red, cv.COLOR_BGR2HSV)
print( hsv_red )      '''
    
# Function that applies dilation to improve grabbing
def dilate(img):
    img = HSVthresholding(img)
    # Apply a Kernel 30x30
    kernel = np.ones((5,5), np.uint8)
    dilation = cv.dilate(img, kernel, iterations = 1)
    
    return dilation
    
# Function that applies morphological operations (Opening) to improve grabbing
def morphOp_Opening(img):
    img = HSVthresholding(img)
    # Apply a Kernel 30x30
    kernel = np.ones((5,5), np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    
    return opening

# --------------------- PART 2: Object Detection ----------------------- #

''' Missing: Add colors to the detected edges'''

# Function that applies a Sobel Edge Detector
def SobelEdgeDetector(img, sc=1, kernel=5):
    delta = 0
    ddepth = cv.CV_64F
    # Apply Gaussian Filter to smooth the image
    img = Gaussian(img, ksize=15)
    # Turn image into grayscale
    gray = GrayScale(img)
    
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=kernel, scale=sc, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=kernel, scale=sc, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    frame_ = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return frame_

# Function that applies a Hough Circles Transform
def HoughCirles(img, param1=100, param2=30, minRadius=1, maxRadius=30):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    gray = Gaussian(gray, ksize=5)
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=param1, param2=param2,
                              minRadius=minRadius, maxRadius=maxRadius)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            #circle center
            cv.circle(img, center, 1, (0,100,100), 3)
            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)
            
    return img


# Function for a Template Matching operation
# It returns the original image with a frame around the detected object
def MatchingOperation(img, templ='template.png', thres=0.9):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    template = cv.imread(templ, 0)
    w, h = template.shape[::-1]
    method = cv.TM_SQDIFF_NORMED
    
    # Apply template Matching
    res = cv.matchTemplate(gray,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF, take minimum
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    
    threshold = thres
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
  
    return img

# Function for a Template Matching operation
# It returns the a grayscale image, where white values indicate higher probability 
    # of detecting the template in a specific location
def MatchingOperationRes(img, templ='template.png'):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    template = cv.imread(templ, 0)
    w, h = template.shape[::-1]
    method = cv.TM_SQDIFF_NORMED
    
    # Apply template Matching
    res = cv.matchTemplate(gray,template,method)
    
    return res
    

# ----------------------- PART 3: Carte Blanche ------------------------ #

# Function that runs a Good Features to Track operation
# it finds the N strongest features in the image by Shi-Tomasi method 
def GoodFeaturesToTrack(img, N=100):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray,N,0.02,20)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv.circle(img, (x,y), 3, 255, -1)
        
    return img


# Creating Bounding boxes and circles for contours
rng.seed(12345)

def ContourBounds(img, thres=80):
    # Turn image into grayscale and apply bluring
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = Bilateral(gray, ksize=5)
    
    # Set threshold for Canny edge detection
    threshold = thres # set hyperparameters
    # Detect edges using Canny
    canny_output = cv.Canny(gray, threshold, threshold*2)
    
    # Find contours
    image, contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    
    # Draw contours + bonding rects + circles
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), 
                     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+ boundRect[i][3])), color, 2)
        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        
    return drawing


# Function that converts the initial video to cartoon
def Cartoonizer(img):
    
    # Apply initially a bilateral filter to reduce the color palette of the video frame.
    
    num_down = 2       # number of downsampling steps
    num_bilateral = 7  # number of bilateral filtering steps
    
    # Downsample the RGB frame using Gaussian pyramid
    img_color = img
    for _ in range(num_down):
        img_color = cv.pyrDown(img_color)
        
    # Apply small Bilateral filter num_bilateral times
    for _ in range(num_bilateral):
        img_color = cv.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        
    # Upsample frame to original size
    for _ in range(num_down):
        img_color = cv.pyrUp(img_color)
         
    # Convert the frame to grayscale and apply median filtering
    blurred = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    # Reduce noise in the initial video frames, using a median filter   
    img_blur = cv.medianBlur(blurred, 7)

    # Detect and enhance edges
    img_edge = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, blockSize=9, C=2)    

    #Combine the two frames
    img_edge = cv.cvtColor(img_edge, cv.COLOR_GRAY2RGB)
    img_cartoon = cv.bitwise_and(img_color, img_edge)
       
    
    return img_cartoon

# ------------------------ PROCESS VIDEO FRAMES ------------------------ #

while(cap.isOpened()):
    # Capture frame by frame
    ret, frame = cap.read()
   
    if not ret:
        break
    
    # Get the frame values in milliseconds
    retval = cap.get(cv.CAP_PROP_POS_MSEC)
    
    
    if ret == True:
        
        # PART 1: Basic Image Processing: 0s - 20s
        
        # -- Switch between color and grayscale
        if retval < 1000 or (retval >= 2000 and retval < 3000):             # sec 0-1 and 2-3
            frame = GrayScale(frame)
        elif (retval >= 1000 and retval < 2000) or (retval >= 3000 and retval < 4000): # sec 1-2 and 3-4
            frame = frame
        
        # -- Smoothing / Bluring
        elif (retval >= 4000 and retval < 6000):
            frame = Gaussian(frame, ksize=15)
        elif (retval >= 6000 and retval < 8000):
            frame = Gaussian(frame, ksize=31)
        elif (retval >= 8000 and retval < 10000):
            frame = Bilateral(frame, ksize=15)
        elif (retval >= 10000 and retval < 12000):
            frame = Bilateral(frame, ksize=31)
        
        # -- Threshold and Morphological Operations
        elif (retval >= 12000 and retval < 14000):
            frame = RBGthresholding(frame)
        elif (retval >= 14000 and retval < 16000):
            frame = HSVthresholding(frame)
        elif (retval >= 16000 and retval < 18000):
            frame = dilate(frame)
        elif (retval >= 18000 and retval < 20000):
            frame = morphOp_Opening(frame)
        
        # PART 2: Object Detection: 20s - 40s
        
        # -- Sobel Edge detector
        elif (retval >= 20000 and retval < 21000):
            frame = SobelEdgeDetector(frame, sc=4, kernel=5)
        elif (retval >= 21000 and retval < 22000):
            frame = SobelEdgeDetector(frame, sc=3, kernel=5)
        elif (retval >= 22000 and retval < 23500):
            frame = SobelEdgeDetector(frame, sc=1, kernel=5)    
        elif (retval >= 23500 and retval < 25000):
            frame = SobelEdgeDetector(frame, sc=1, kernel=3)
        
        # -- Hough Circles Transform
        elif (retval >= 25000 and retval < 27000):
            frame = HoughCirles(frame, param1=150, param2=30, minRadius=1, maxRadius=0)
        elif (retval >= 27000 and retval < 29000):
            frame = HoughCirles(frame, param1=180, param2=30, minRadius=1, maxRadius=0)
        elif (retval >= 29000 and retval < 31000):
            frame = HoughCirles(frame, param1=180, param2=30, minRadius=1, maxRadius=30)    
        elif (retval >= 31000 and retval < 33000):
            frame = HoughCirles(frame, param1=180, param2=40, minRadius=1, maxRadius=40)
        
        
        # -- Template Matching
        elif (retval >= 35000 and retval < 40000):
            frame = MatchingOperation(frame, templ='template.png', thres=0.9)
        
        # PART 3: Carte blanche: 40s - 60s
        
        elif (retval >= 40000 and retval < 50000):
            frame = ContourBounds(frame)
        elif (retval >= 50000 and retval < 55000):
            frame = GoodFeaturesToTrack(frame, N=100)
        else:
            frame = Cartoonizer(frame) 
          
        
        # Display processed video
        cv.imshow('Frame',frame)
        
        # Write processed video
        out.write(frame)
       	
        # Press Q on keyboard to exit
       	if cv.waitKey(25) & 0xFF == ord('q'):
       		break 


# Release the processed video
cap.release()

# Release the video capture object
out.release()

# Closes all the frames
cv.destroyAllWindows()

# That's all Folks
# ps. It was a fun assignment!!!
