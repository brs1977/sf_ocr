import tempfile
import shutil
import fitz
import io
import re
import os
import json
from PIL import Image
import numpy as np
from fitz import fitz, Rect
from img_utils import *

INDEX_PATTERN = re.compile('_(\d*)')


def temp_file_name():
    return next(tempfile._get_candidate_names())


class PDFSplitter:
    def __init__(self, out_file_name, pdf_file, orient_clf, type_clf, extractor):
        self.files = {}
        self.out_file_name = out_file_name
        self.pdf_file = pdf_file
        self.orient_clf = orient_clf
        self.type_clf = type_clf
        self.extractor = extractor

    
    def pdf_page_image_gen(self, file_name):
        with fitz.open(file_name) as pdf_file:
            pages = len(pdf_file)
            for page_index in range(pages):
                page = pdf_file[page_index]

                # # get text block
                # blocks = page.get_text("blocks")
                # blocks.sort(key=lambda block: block[3])  # sort by 'y1' values
                # text = [ ' '.join(block[4].split()) for block in blocks]

                # max image
                images = [pdf_file.extract_image(img[0]) for img in page.get_images() ]      
                images.sort(key=lambda img : -img['width']*img['height'] )

                yield images[0], page_index+1, pages
    
    
    def image_to_pdf_page(self, pdf_new, img):
        newpage = pdf_new.new_page(-1, width=img.shape[1], height=img.shape[0])
        rect = Rect(0, 0, img.shape[1], img.shape[0])
        img = Image.fromarray(img).convert('RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='jpeg')
        newpage.insert_image(rect, stream=img_bytes.getvalue())

    def preprocess_image(self, image):
        # if image['width'] + image['height'] < 1400:
        #   return -1, None

        # # get the image extension
        # image_ext = image["ext"]

        image_bytes = image["image"]
        img = np.asarray(Image.open(io.BytesIO(image_bytes)))

        gray = to_gray(img)
        h, w = gray.shape
        angle = 0
        # set landsape
        if h > w:
            gray = np.rot90(gray, 1)
            angle = 90

        gray = resize(gray, 140, 100)
        orient = self.orient_clf.predict([gray])[0]

        if orient == 1:
            angle -= 180
            gray = np.rot90(gray, 2)

        typ = self.type_clf.predict([gray])[0]

        img = np.rot90(img, angle//90)
        _, img = correct_skew3(img)

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

    def create_pdf(self, tmp_dir, imgs):
        if not imgs:
            return None

        info, text = self.extractor.process(imgs[0])
        # logger.debug(text)
        # logger.debug(info)

        file_name = temp_file_name()  # self._get_file_name(info['sf_no'])

        # with open(os.path.join(tmp_dir,file_name+'.json'), 'w') as json_file:
        #   json_file.write(json.dumps(info))

        files = []
        for i, img in enumerate(imgs):
            with fitz.open() as pdf_new:
                self.image_to_pdf_page(pdf_new, img)
                fn = f'{file_name}-{i+1}.pdf'
                files.append(fn)
                pdf_new.save(os.path.join(tmp_dir, fn),
                             garbage=4, deflate=True)

        info['files'] = files
        return info

    def process(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = []
            imgs = []
            is_first = True
            for image, page, pages in self.pdf_page_image_gen(self.pdf_file):
                typ, img = self.preprocess_image(image)

                if typ == 1 and not is_first:
                    info = self.create_pdf(tmp_dir, imgs)
                    results.append(info)
                    imgs = []
                    yield page, pages, info

                imgs.append(img)
                is_first = False

            info = self.create_pdf(tmp_dir, imgs)
            results.append(info)

            with open(os.path.join(tmp_dir, 'results.json'), 'w') as json_file:
                json_file.write(json.dumps(results))

            # zip
            shutil.make_archive(self.out_file_name, 'zip', tmp_dir)

            yield page, pages, info

