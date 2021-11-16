import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from loguru import logger
from config import Config
from extractor import SfInfoExtractor
from hog_classifier import load_model
from tqdm import tqdm
from img_utils import *
from pdf_utils import *

logger.add('logs/ds.log')

config = Config('models/config.yaml')
extractor = SfInfoExtractor(config)
orient_clf = load_model('models/orient.pkl')
type_clf = load_model('models/type.pkl')



def file_info(file_name):
  typ, orient = os.path.basename(file_name).split('_')[-2:]
  typ = 1 if typ == 'sf' else 0
  orient = int(orient[0])
  return typ, orient


def extract(file_name):
    typ, orient = file_info(file_name)
    if typ != 1:
        raise Exception('no sf')
        
    img = cv2.imread(file_name)
    if orient==1:
        img = np.rot90(img,2)
    

    info, text = extractor.process(img)
    file_name = os.path.basename(file_name)
    info['file_name'] = file_name
    info['typ'] = typ
    info['orient'] = orient
    info['text'] = text
    return info


def pdf_to_ds():
    n = 200

    files = glob('../pdf/*.pdf')
    for pdf_file in tqdm(files):
        for img, page, pages in img_page_pdf_gen(pdf_file):    

            h,w = img.shape[:2]
            if h+w < 1400: # min 800x600
                continue
            gray = to_gray(img)  
            angle = 0
            # set landsape
            if h > w:
                gray = np.rot90(gray, 1)
            angle = 90

            gray = resize(gray, 140,100)    
            orient_pred = orient_clf.predict([gray])[0]

            if orient_pred == 1:
                angle -= 180
                gray = np.rot90(gray, 2)

            type_pred = type_clf.predict([gray])[0]

            sf = '' if type_pred == 1 else '1'
            file_name = os.path.basename(pdf_file).split('.')[0]
            file_name = f'../imgs/{file_name}_{page}_sf{sf}_0.png'   

            img = np.rot90(img,angle//90)
            gray = to_gray(img)  
            cv2.imwrite(file_name,gray)




def img_to_text():
    data = []
    for i, file_name in enumerate(glob('dataset/png/*.png')):
        try:
            logger.debug('{} {}',i,file_name)
            info = extract(file_name)
            data.append(info)
        except Exception as e:
            logger.error(e)


    df = pd.DataFrame(data)
    df.to_csv('output/csv/ocr11_6.csv',index=False)

img_to_text()     