import os
import io
import fitz
from loguru import logger

# from config import Config
# from extractor import SfInfoExtractor
# from hog_classifier import load_model
# from pdf_splitter import PDFSplitter

MIN_IMAGE_SIZE  = 600 * 400

def split():
    file_path = 'output'
    config = Config('models/config.yaml')
    extractor = SfInfoExtractor(config)
    orient_clf = load_model('models/orient.pkl')
    type_clf = load_model('models/type.pkl')

    pdf_file_name = '.\\input\\1693+1694+02092024.pdf'
    zip_file_name = '.\\output'+os.path.basename(pdf_file_name)+'.zip'

    splitter = PDFSplitter(zip_file_name, pdf_file_name,orient_clf, type_clf, extractor)

    results = []
    for page, pages, info in splitter.process():            
        results.append(info)
    print(results)    

def pdf_page_text(page):
    # get text block
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda block: block[3])  # sort by 'y1' values
    text = [ ' '.join(block[4].split()) for block in blocks]
    return '\n'.join(text)


def max_page_image(pdf_file, page):
  # max image by width * height
  #get_page_images() (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer)
#   images = pdf_file.get_page_images(page)
#   return max(images, key = lambda x: x[2]*x[3] ) 

  images = pdf_file[page].get_image_info(xrefs=True)
  image = max(images, key = lambda x: x['width']*x['height'] )   
  return image

def pdf_is_text(file_name):  
  with fitz.open(file_name) as pdf_file:
    if not pdf_file.is_pdf:
        raise ValueError('File is not pdf') 
    if len(pdf_file) == 0:
        raise ValueError('Pdf has not pages') 
    if len(pdf_file[0].get_image_info(xrefs=True)) == 0:
        logger.debug('Text pdf page')  
        return True

    image = max_page_image(pdf_file, 0)
    if image['width'] * image['height'] < MIN_IMAGE_SIZE:
        logger.debug('Text pdf page')  
        return True    
    
    # for page in pdf_file:
    #    print(pdf_page_text(page))
  return False

def mirror_test(file_name):
  from PIL import Image
  with fitz.open(file_name) as pdf_file:
      pages = len(pdf_file)
      for page_index in range(pages):
        images = pdf_file[page_index].get_image_info(xrefs=True)
        image_info = max(images, key = lambda x: x['width']*x['height'] )
        xref = image_info['xref']
        image = pdf_file.extract_image(xref)            

        image_bytes = image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image.save('out.png')
   

if __name__ == '__main__':
  #  file_name = '.\\input\\1693+1694+02092024.pdf'
  # file_name = './input/ттттттт.pdf'
  # file_name = './input/2025+2024+2023+2022+2038+2031+2030+2029+2028+2027+2026+2035+20_12.pdf'
  file_name = './input/390.pdf'
  # print(pdf_is_text(file_name))
  mirror_test(file_name)
