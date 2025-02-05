import tempfile
import shutil
import fitz
import io
import re
import os
import json
from loguru import logger
from PIL import Image
import numpy as np
from fitz import Rect
from img_utils import to_gray, resize, correct_skew3

INDEX_PATTERN   = re.compile('_(\d*)')
MIN_IMAGE_SIZE  = 600 * 400

def temp_file_name():
    return next(tempfile._get_candidate_names())

def max_file_page_image(pdf_file, page):
  # max image by width * height
  #get_page_images() (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer)
#   images = pdf_file.get_page_images(page)
#   return max(images, key = lambda x: x[2]*x[3] ) 

  images = pdf_file[page].get_image_info(xrefs=True)
  image = max(images, key = lambda x: x['width']*x['height'] )   
  return image

def max_page_image(page):
  images = page.get_image_info(xrefs=True)
  image = max(images, key = lambda x: x['width']*x['height'] )   
  return image

def is_text_page(page):
    if len(page.get_image_info(xrefs=True)) == 0:
        logger.debug('Text pdf page')  
        return True

    image = max_page_image(page)
    if image['width'] * image['height'] < MIN_IMAGE_SIZE:
        logger.debug('Text pdf page')  
        return True    
    return False

def pdf_is_text(file_name):  
  with fitz.open(file_name) as pdf_file:
    if not pdf_file.isPDF:
        raise ValueError('File is not pdf') 
    if len(pdf_file) == 0:
        raise ValueError('Pdf has not pages') 
    if len(pdf_file[0].get_image_info(xrefs=True)) == 0:
        logger.debug('Text pdf page')  
        return True

    image = max_file_page_image(pdf_file, 0)
    if image['width'] * image['height'] < MIN_IMAGE_SIZE:
        logger.debug('Text pdf page')  
        return True    
  return False

def pdf_page_text(page):
    # get text block
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda block: block[3])  # sort by 'y1' values
    text = [ ' '.join(block[4].split()) for block in blocks]
    return '\n'.join(text)

def pdf_page_text_gen(file_name):
    with fitz.open(file_name) as pdf_file:
        pages = len(pdf_file)
        for page_index in range(pages):
            page = pdf_file[page_index]

            text = pdf_page_text(page)

            yield text, page_index+1, pages


def pdf_page_image_gen(file_name, pages_range):
    with fitz.open(file_name) as pdf_file:
        pages = len(pdf_file)
        for page_index in pages_range:
            image_info = max_file_page_image(pdf_file, page_index)
            xref = image_info['xref']
            image = pdf_file.extract_image(xref)   

            image_bytes = image["image"]    
            raw_image = Image.open(io.BytesIO(image_bytes))
            print(image_info)
            if 'transform' in image_info.keys() and image_info['transform'][3] < 0: # Проверка на зеркальное отображение TODO надобы image_info['transform'][2] сравнить на -0.0
                raw_image = raw_image.transpose(Image.FLIP_LEFT_RIGHT)
                logger.debug('Зеркальное отображение FLIP_LEFT_RIGHT')

            image = np.asarray(raw_image)
            yield image, page_index+1, pages

def pdf_image_to_page(pdf_new, img):
    newpage = pdf_new.new_page(-1, width=img.shape[1], height=img.shape[0])
    rect = Rect(0, 0, img.shape[1], img.shape[0])
    img = Image.fromarray(img).convert('RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='jpeg')
    newpage.insert_image(rect, stream=img_bytes.getvalue())

def pdf_bytes_to_image(image):
    image_bytes = image["image"]    
    raw_image = Image.open(io.BytesIO(image_bytes))
    return np.asarray(raw_image)

def pdf_create_page_file(filename, docsrc, from_page):
  with fitz.open() as pdf_new:
    pdf_new.insert_pdf(docsrc, from_page=from_page, to_page=from_page)
    pdf_new.save(filename, garbage=4, deflate=True)

class PDFSplitter:
    def __init__(self, out_file_name, pdf_file, orient_clf, type_clf, extractor):
        self.files = {}
        self.out_file_name = out_file_name
        self.pdf_file = pdf_file
        self.orient_clf = orient_clf
        self.type_clf = type_clf
        self.extractor = extractor

    def preprocess_image(self, img):        
        gray = to_gray(img)
        h, w = gray.shape
        angle = 0
        # set landsape
        if h > w:
            gray = np.rot90(gray, 1)
            angle = 90

        gray = resize(gray, 140, 100)
        orient = self.orient_clf.predict([gray])[0]
        logger.debug(('orient', orient))

        if orient == 1:
            angle -= 180
            gray = np.rot90(gray, 2)

        typ = self.type_clf.predict([gray])[0]
        logger.debug(('type', typ))

        img = np.rot90(img, angle//90)
        angle, img = correct_skew3(img)
        logger.debug(('skew', angle))

        return typ, img

    def _get_file_name(self, fn):

        if not fn:
            fn = 'undefined'
        else:
            fn = fn.replace('/', '_')
        file_name = fn

        while fn in self.files.keys():
            idx = INDEX_PATTERN.findall(fn)
            if idx:
                idx = idx[-1]
            if not idx:
                idx = 0
            idx = int(idx) + 1
            fn = file_name + '_' + str(idx)

        self.files[fn] = fn
        return fn

    def pdf_create_file(self, tmp_dir, imgs):
        if not imgs:
            return None

        info, text = self.extractor.process(imgs[0])
        # logger.debug(text)
        # logger.debug(info)

        file_name = temp_file_name()  

        # with open(os.path.join(tmp_dir,file_name+'.json'), 'w') as json_file:
        #   json_file.write(json.dumps(info))

        files = []
        for i, img in enumerate(imgs):
            with fitz.open() as pdf_new:
                pdf_image_to_page(pdf_new, img)
                fn = f'{file_name}-{i+1}.pdf'
                files.append(fn)
                pdf_new.save(os.path.join(tmp_dir, fn),
                             garbage=4, deflate=True)

        info['files'] = files
        return info

    def process_image_pdf(self, tmp_dir, pages_range):
        imgs = []
        is_first = True
        for image, page, pages in pdf_page_image_gen(self.pdf_file, pages_range):
            typ, img = self.preprocess_image(image)

            if typ == 1 and not is_first:
                info = self.pdf_create_file(tmp_dir, imgs)
                imgs = []
                yield info

            imgs.append(img)
            is_first = False

        info = self.pdf_create_file(tmp_dir, imgs)
        yield info

    def is_first_page(self, text):
        # тест 1 страницы документа, со словами инн/кпп и фактура
        text = text.upper()
        return self.extractor.config.PATTERN_INN_KPP.search(text) and  self.extractor.config.PATTERN_SF_NUM.search(text)
            

    def process_text_pdf(self, tmp_dir, pages_range):
        info = None
        with fitz.open(self.pdf_file) as pdf_file:
            page_no = 1
            pages = len(pdf_file)
            for page_index in pages_range:                
                text = pdf_page_text(pdf_file[page_index])                                    
                
                if self.is_first_page(text):                    
                    if info: 
                        yield info

                    page_no = 1
                    file_name = temp_file_name()  

                    info = self.extractor.extract_sf_data(text)
                    try:
                        if not info['buyer_inn'] and not info['seller_inn']:                            
                            info = self.extractor.extract_sf_data_from_text_pdf(text)
                    except Exception as e:
                        logger.debug(e)
                    
                    info['files'] = []
                    logger.debug(info)                        


                fn = f'{file_name}-{page_no}.pdf'
                info['files'].append(fn)
                fn = os.path.join(tmp_dir, fn)
                pdf_create_page_file(fn, pdf_file, page_index)
                page_no += 1

            # info['files'] = files
            # results.append(info)
            # logger.debug(info)
            yield info

    def process(self):
        with fitz.open(self.pdf_file) as pdf_file:
            if not pdf_file.is_pdf:
                raise ValueError('File is not pdf') 
            if len(pdf_file) == 0:
                raise ValueError('Pdf has not pages') 

            pages = len(pdf_file)
            text_pages_range = []
            image_pages_range = []
            for page_index in range(pages):
                if is_text_page(pdf_file[page_index]):
                    text_pages_range.append(page_index)
                else:
                    image_pages_range.append(page_index)

        with tempfile.TemporaryDirectory() as tmp_dir:
            results = []
            page = 0
            pages = len(text_pages_range) + len(image_pages_range)
            for info in self.process_text_pdf(tmp_dir, text_pages_range):
                page += 1
                if info:
                    results.append(info)
                yield page, pages, info
            for info in self.process_image_pdf(tmp_dir, image_pages_range):
                page += 1
                if info:
                    results.append(info)
                yield page, pages, info

            with open(os.path.join(tmp_dir, 'results.json'), 'w') as json_file:
                json_file.write(json.dumps(results))

            # zip
            shutil.make_archive(self.out_file_name, 'zip', tmp_dir)        



        # if pdf_is_text(self.pdf_file):
        #     for x in self.process_text_pdf(text_pages_range):
        #         yield x
        # else:
        #     for x in self.process_image_pdf(image_pages_range):
        #         yield x


            

