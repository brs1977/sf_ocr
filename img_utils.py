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
from skimage.util import img_as_float, img_as_ubyte
from skimage.morphology import skeletonize

# pytesseract.pytesseract.tesseract_cmd = r'/home/ruslan/prj/sf_ocr/tesseract/tesseract'
# os.environ['TESSDATA_PREFIX'] = r'/home/ruslan/prj/sf_ocr/tesseract/tessdata'


def dpi(w, h, tw=11.75, th=8.25):
  '''
  Where DPI is the average dots per inch
  W is the total horizontal pixels
  TW is the total width (in)
  H is the total vertical pixels
  TH is the total length (in)
  A4 8-1/4 x 11-3/4 in

  >>> dpi(3508,2480) = 300
  '''
  return round((w/tw + h/th ) /2)

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


def rotation(img,angle,background=(255,255,255)):
  old_width, old_height = img.shape[:2]
  
  angle_radian = math.radians(angle)
  width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
  height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

  image_center = (old_height/2, old_width/2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  rot_mat[1, 2] += (width - old_width) / 2
  rot_mat[0, 2] += (height - old_height) / 2
  return rot_mat,cv2.warpAffine(img, rot_mat, (int(round(height)), int(round(width))), borderValue=background)  



def point_rotation(points,rot_mat):
  points = np.array(points)
  # add ones
  ones = np.ones(shape=(len(points), 1))

  points_ones = np.hstack([points, ones])

  # transform points
  transformed_points = rot_mat.dot(points_ones.T).T

  return np.asarray(transformed_points, dtype=np.uint16)

def table_roi(mask):
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contour_areas = list(map(cv2.contourArea, contours))
  largest_contour_idx = np.argmax(list(contour_areas))
  contour = contours[largest_contour_idx]
  # Contour Approximation # меньше epsilon более точно повторяется контур
  epsilon = 0.09*cv2.arcLength(contour,True)
  approx = cv2.approxPolyDP(contour,epsilon,True)
  x,y,w,h = cv2.boundingRect(approx)

  # корректировка если регион таблицы не найден
  h = h if h > mask.shape[0] / 10 else mask.shape[0]
  w = w if w > mask.shape[1] / 2 else mask.shape[1]

  return x,y,w,h

def calculate_angle(boxes):
  img_edges = img_as_ubyte(skeletonize(img_as_float(boxes)))
  lines = cv2.HoughLinesP(img_edges, rho=1, theta=np.pi / 180.0, threshold=160, minLineLength=150, maxLineGap=10)
  # lines = cv2.HoughLinesP(img_edges, 1, 1 / 180.0, 100, minLineLength=100, maxLineGap=10)

  angles = []

  try:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
          if angle != 0 and angle != -90:
              if angle > 30:
                  angle = 90 - angle
              if angle < -30:
                  angle = angle + 90
              angles.append(angle)

      if len(angles) != 0:
          skew_angle = np.mean(angles)
      else:
          skew_angle = 0
  except:
      skew_angle = 0

  return skew_angle