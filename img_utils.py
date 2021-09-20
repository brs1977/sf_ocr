import cv2
import re
import os
from PIL import Image
from glob import glob
import io
from loguru import logger
import math
import fitz
import pytesseract as pytesseract
import numpy as np
# from google.colab.patches import cv2_imshow
import functools

def resize(img, x,y):
  if img.shape[0]>img.shape[1]:
    size = (y, x)
  else:
    size = (x, y)  
  return cv2.resize(img, size)


def apply_threshold(img, method, kernel_size = 1):
  # =============================================================================== #
  #    Threshold Methods                                                            #
  # =============================================================================== #
  # 1. Binary-Otsu                                                                  #
  # 2. Binary-Otsu w/ Gaussian Blur (kernel size, kernel size)                      #
  # 3. Binary-Otsu w/ Median Blur (kernel size, kernel size)                        #
  # 4. Adaptive Gaussian Threshold (31,2) w/ Gaussian Blur (kernel size)            #
  # 5. Adaptive Gaussian Threshold (31,2) w/ Median Blur (kernel size)              #
  # =============================================================================== #  
    switcher = {
        1: cv2.threshold(img, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (kernel_size, kernel_size), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.medianBlur(img, kernel_size), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (kernel_size, kernel_size), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        5: cv2.adaptiveThreshold(cv2.medianBlur(img, kernel_size), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(method, "Invalid method")

def to_gray(src):
  return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  if len(src.shape)==3 else src.copy()

def to_binary(gray):
  return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

def to_lines(bw, structure):
  # Apply morphology operations
  mat = cv2.erode(bw, structure, (-1, -1))
  mat = cv2.dilate(mat, structure, (-1, -1))
  return mat  

def preprocess_image(img, method=1, kernel_size=5):
    gray = to_gray(img)
    gray = apply_threshold(gray, method=method)

    # dilate the text to make it solid spot
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    gray = cv2.dilate(~gray, struct, anchor=(-1, -1), iterations=1)
    return gray   
  
def get_contours(mask):
  # find contour min x max x and min y max y
  contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  contour_areas = list(map(cv2.contourArea, contours))

  if not contour_areas:
    return contour_areas, 0, 0, 0, 0

  largest_contour_idx = np.argmax(contour_areas)
  largest_contour = contours[largest_contour_idx]

  x = [x[0][0] for x in largest_contour]
  y = [x[0][1] for x in largest_contour]

  min_x, max_x = min(x), max(x)
  min_y, max_y = min(y), max(y)

  return contours,min_x,min_y,max_x,max_y

def correct_skew3(image, background = (255,255,255)):
  orig = image.copy()
  
  if len(image.shape)==3:
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

  img_edges = cv2.Canny(image, 200, 200, apertureSize=3)
  lines = cv2.HoughLinesP(img_edges, rho=1, theta=np.pi / 180.0, threshold=160, minLineLength=150, maxLineGap=10)
  
  # calculate all the angles:
  angles = []
  for [[x1, y1, x2, y2]] in lines:
    # Drawing Hough lines
    # cv2.line(image, (x1, y1), (x2, y2), (128,0,0), 30)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)
    # if 10 > angle > -10:
    #   angles.append(angle)
    
  # average angles
  median_angle = np.median(angles)
  # actual rotation
  # image = ndimage.rotate(image, median_angle)
  # return median_angle, image

  # cv2_imshow(image)

  old_width, old_height = image.shape
  angle_radian = math.radians(median_angle)
  width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
  height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

  image_center = tuple(np.array(image.shape[::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, median_angle, 1.0)
  rot_mat[1, 2] += (width - old_width) / 2
  rot_mat[0, 2] += (height - old_height) / 2
  return median_angle, cv2.warpAffine(orig, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def ocr(img, lang, config):
  return pytesseract.image_to_string(img, lang=lang, config=config)

def ocr_rus(img, lang='rus', config='--oem 1 --psm 4'):
  '''
        psm 4    Assume a single column of text of variable sizes.
  '''
  text = ocr(img, lang, config)
  # logger.debug(text)
  return text



