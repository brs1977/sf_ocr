from glob import glob
import fitz
import io
import os
import cv2
import math
import json
from PIL import Image
import numpy as np
# from google.colab.patches import cv2_imshow
from skimage.feature import hog, local_binary_pattern
from fitz import fitz, Rect
from img_utils import *


def img_page_pdf_gen(file_name):  
  pdf_file = fitz.open(file_name)
    
  # iterate over PDF pages
  for page_index in range(len(pdf_file)):          
    # get the page itself
    page = pdf_file[page_index]
    
    image_list = page.getImageList()

    for image_index, img in enumerate(page.getImageList(), start=1):
        
      # get the XREF of the image
      xref = img[0]
        
      # extract the image bytes
      base_image = pdf_file.extractImage(xref)
      image_bytes = base_image["image"]
      
      # get the image extension
      image_ext = base_image["ext"]
      # print(base_image["ext"], base_image["smask"], base_image["bpc"], base_image["cs-name"])
      img = np.asarray(Image.open(io.BytesIO(image_bytes)))
      # print(img.shape)
      # cv2_imshow(img)

      logger.info(f"{{'page':{page_index+1}, 'pages':{len(pdf_file)}}}")
      yield img
  

def create_image_page(pdf_new, img):
  newpage = pdf_new.newPage(-1, width=img.shape[1], height=img.shape[0])
  rect = Rect(0, 0, img.shape[1], img.shape[0])
  img = Image.fromarray(img).convert('RGB')
  img_bytes = io.BytesIO()
  img.save(img_bytes, format='jpeg')
  newpage.insertImage(rect, stream=img_bytes.getvalue())

def split_pdf(pdf_file, orient_clf, type_clf, extractor, path='output'):

  basename = os.path.basename(pdf_file)
  path = os.path.join(path, basename.split('.')[0])
  os.makedirs(path, exist_ok=True)

  imgs = []
  is_first = True
  n = 1
  for img in img_page_pdf_gen(pdf_file):
    gray = to_gray(img)
    h,w = gray.shape
    angle = 0
    # set landsape
    if h > w:
      gray = np.rot90(gray, 1)
      angle = 90

    gray = resize(gray, 140,100)    
    orient = orient_clf.predict([gray])[0]

    if orient == 1:
      angle -= 180
      gray = np.rot90(gray, 2)

    typ = type_clf.predict([gray])[0]

    img = np.rot90(img,angle//90)    
    if typ == 1 and not is_first:
      create_pdf_file(imgs, path, extractor)
      n += 1
      imgs = []

    _, img = correct_skew3(img)
    imgs.append(img)

    # img = resize(img, 350, 200)
    # print(typ)
    # cv2_imshow(img)

    is_first = False  

  create_pdf_file(imgs, path, extractor)

def get_file_index(path):
  files = glob(path + '*.pdf')
  if len(files) == 0:
    return ''
  return '_' + str(len(files))

def get_file_name(fn, path):
  if not fn:
    fn = 'undefined'
  else:
    fn = fn.replace('/','_')
  index = get_file_index(os.path.join(path,fn))
  return fn + index

def create_pdf_file(imgs, path, extractor):

  # text = text_from_img(imgs[0])    
  # info = sf_info_from_img_text(text.split('\n'))        
  info = extractor.process(imgs[0])
  # logger.debug(text)
  logger.debug(info)

  file_name = get_file_name(info['sf_no'], path)

  pdf_new = fitz.open()

  for img in imgs:
    create_image_page(pdf_new, img)  

  file_name = os.path.join(path,file_name)
  logger.debug('save ' + file_name + '.pdf')  
  pdf_new.save(file_name + '.pdf', garbage=4, deflate=True)  

  with open(file_name + '.json', 'w') as f:
    json.dump(info, f)
