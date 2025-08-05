import fitz
import io
import re
import os
import cv2
from PIL import Image, ImageOps
import numpy as np
from fitz import Rect
from pdf_splitter import generate_filename, save_image


def pdf_bytes_to_image(image):
    img_data = image["image"]
    img = Image.open(io.BytesIO(img_data))
    
    # Инвертируем все бинарные изображения (1 бит на пиксель)
    if image.get('bpc', 0) == 1:
        # Проверяем цветовое пространство (числовое или строковое)
        colorspace = image.get('colorspace')
        is_grayscale = (colorspace in [1, 'DeviceGray', 'Gray']) or (isinstance(colorspace, int) and colorspace == 1)
        
        print(img.mode, img.mode == '1', is_grayscale)

        if is_grayscale and img.mode == 'L':
            img = ImageOps.invert(img)
    
    # Конвертируем бинарные изображения в 8-битные серые
    if img.mode == '1':
        img = img.convert('L')
    
    return np.asarray(img)

file_name = 'input/873b6d73-8705-4f8f-8ae8-8bc4dd08d43e.pdf'
with fitz.open(file_name) as pdf_file:
    if not pdf_file.isPDF:
        raise ValueError('File is not pdf') 
    if len(pdf_file) == 0:
        raise ValueError('Pdf has not pages') 

    pages = range(len(pdf_file))

    for page_index in pages:
        pix = pdf_file[page_index].get_pixmap()

        # Преобразуем в numpy массив с нужной формой
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        # Если надо убрать альфа-канал (если есть)
        if pix.n == 4:
            img = img[:, :, :3]  # Оставляем только RGB        
        save_image(generate_filename(), img)        

        # xref = max_file_page_image(pdf_file, page_index)['xref']
        images = pdf_file[page_index].get_image_info(xrefs=True)
        # print(dir(pdf_file[page_index]))
        # image = max(images, key = lambda x: x['width']*x['height'] )   
        # image = pdf_file.extract_image(xref)

        for xref in images:
            print(xref)
            image = pdf_file.extract_image(xref['xref'])
            image_bytes = image["image"]
            img = np.asarray(Image.open(io.BytesIO(image_bytes)))

            save_image(generate_filename(), img)