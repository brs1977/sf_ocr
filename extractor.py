import re
import cv2
import numpy as np
from img_utils import ocr_data, ocr_rus, ocr, dpi, preprocess_image, to_binary, get_contours, to_lines, table_roi, calculate_angle, rotation
import functools
from loguru import logger

class NoSfException(Exception):
    pass

def clean_space(text):
  return ''.join([x for x in text if x!=' '])  

def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.log(level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs)
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper


def regexp_group_by_pattern(arr,pattern):
  for s in arr:
    grp = pattern.findall(s.upper())
    if grp:
      return grp[0][1]
  return None

def group_by_pattern(pattern,text,s=0):
  for i,t in enumerate(text[s:]):
    grp = pattern.findall(text[s+i].upper())
    if grp:
      if grp[0][1]:
        return i+s, grp[0][1]
      else: # если номер стоит впереди 
        return i+s, t
  raise LookupError  

def format_text(details, left_break):
    parse_text = []
    coords = []
    word_list = []
    last_word = ''
    left = 10000 
    top = 10000 
    width = height = 0
    for i, word in enumerate(details['text']):
        if details['top'][i] == 0 and details['left'][i] == 0:
          continue
        if word != '':                         
            if details['left'][i] < left_break:
              top = min(top,details['top'][i]) 
              left = min(left,details['left'][i]) 
              # height = max(height,details['height'][i])                         
              height = max(height, (details['top'][i] + details['height'][i]) - top)
              width = max(left, (details['left'][i] + details['width'][i]) - left)

              word_list.append(word)
            last_word = word
        if (last_word != '' and word == '') or (word == details['text'][-1]):
            if word_list and word_list != [' ']:
                parse_text.append(' '.join(word_list))
                coords.append((left, top, width, height))

            left = 10000 
            top = 10000 
            width = height = 0

            word_list = []

    return parse_text, coords

def find_part_text(data, part):
  for i, x in enumerate(data['text']):
    if x.upper().startswith(part):
      return i
  raise LookupError

class SfInfoExtractor:
  def __init__(self, config):
    self.config = config

  @logger_wraps()
  def extract_date(self, dt):      
    def to_date(d,m,y):
      for x in self.config.NUMS_LETTER_CORRECT:
        d = d.replace(x,str(self.config.NUMS_LETTER_CORRECT[x]))

      m = f'{int(m):02d}'
      d = f'{int(d):02d}'
      return '.'.join([d,m,y])

    if not dt:
      return None
    
    res = self.config.PATTERN_DATE_SEARCH.search(dt)
    dt = res.groups()[1] if res else dt
    split_data = self.config.PATTERN_DATE_SPLIT.split(dt)    

    if len(split_data)==1 and len(split_data[0])>10:
      return ''
    elif len(split_data)==3:
      split_data = [x.strip() for x in split_data]
      d,m,y = split_data
      d = ''.join([x for x in d if x!='.'])
      m = self.config.MONTH[m]   
      return to_date(d,m,y)

    dt = dt.replace('-','.')
    dt = dt.replace('/','.')

    dt_split = dt.split('.')    
    if len(dt_split)==1:
      return ''
    elif len(dt_split) == 3:
      d, m, y = dt_split
    elif len(dt_split) == 2:
      d, m = dt_split
      if len(m)<=4:        
        y = m[-2:]
        m = m[:-2]
      else:
        y = m[4:]
        m = m[:-4]
    return to_date(d,m,y)


  def extract_sf_no_and_date(self, text):
    def sf_no_and_date(i, sf_no_group, text):
      # # очистка от ошибок распознавания
      # sf_no_group = ''.join([x for x in sf_no_group if x not in ',\[\]`‘\'"~!@#$%^&*();:?*+=|\\'])
            
      # ищем дату, до даты получается номер
      res = self.config.PATTERN_DATE_SEARCH.search(sf_no_group)      
      if res:
        sf_no, sf_date = res.groups()
        sf_no = self.extract_sf_no(sf_no)
        sf_date = self.extract_date(sf_date)
        
      # выделяем номер дата ниже
      else:
        sf_no = self.extract_sf_no(sf_no_group)
        sf_date = ' '.join([t for t in text[i+1:i+3] if 8<len(t)<25])
        if not sf_date.strip():
          sf_date = ' '.join([t for t in text[max(0,i-3):i] if 8<len(t)<25])
        sf_date = self.extract_date(sf_date)
      return sf_no, sf_date

    try:
      try:
        #ищем по патерну с/ф
        i, sf_no_group = group_by_pattern(self.config.PATTERN_SF_NUM,text)  
      except:
        # если счет/фактура разбита на 2 слова, ищем по патерну счет или фактура
        i,sf_no_group = group_by_pattern(self.config.PATTERN_BILL,text)  
        sf_no_group = ' '.join([t for t in text[i+1:i+4] if len(t)<25]) 
        i, sf_no_group = group_by_pattern(self.config.PATTERN_SF_NUM,[sf_no_group])        
      
      if len(sf_no_group.strip()) > 1:
        if len(sf_no_group.strip()) > 10:
          return sf_no_and_date(i, sf_no_group, text)
        else:
          sf_no_group += ' ' + ' '.join([t for t in text[i+1:i+2] if len(t)<25])  
          return sf_no_and_date(i, sf_no_group, text)
      else: # если строка короткая ищем ниже или выше
        sf_no_group = ' '.join([t for t in text[i+1:i+4] if len(t)<25])
        sf_no, sf_date = sf_no_and_date(i, sf_no_group, text)
        if sf_no or sf_date:
          return sf_no, sf_date

        sf_no_group = ' '.join([t for t in text[max(0,i-4):i] if len(t)<25])
        return sf_no_and_date(i, sf_no_group, text)
    except Exception as e:
      logger.exception(e)
      return None, None

  def extract_sf_data_from_text_pdf(self,text):
    text = text.upper()    
    if self.is_no_sf(text):
      raise NoSfException()

    # делим по переводу каретки удаляем пустые строки
    text = [x for x in text.split('\n') if x.strip()]
    text = self.clean_text(text)
    sf_no, sf_date = self.extract_sf_no_and_date(text)

    inn_seller, kpp_seller, inn_buyer, kpp_buyer = None, None, None, None

    # inn_kpp_patern = re.compile('ИНН/КПП')

    # индексы слов
    sid, _ = group_by_pattern(self.config.PATTERN_SELLER,text)
    bid, _ = group_by_pattern(self.config.PATTERN_BUYER,text)

    ik_seller, _ = group_by_pattern(self.config.PATTERN_INN_KPP,text,sid+1)
    ik_buyer, _ = group_by_pattern(self.config.PATTERN_INN_KPP,text,bid+1)


    if sid < bid and ik_seller <  ik_buyer:
      inn_seller, kpp_seller = self.config.PATTERN_INN_KPP.findall(text[ik_seller])[0]
      inn_buyer, kpp_buyer = self.config.PATTERN_INN_KPP.findall(text[ik_buyer])[0]


    return {'sf_no':sf_no, 'sf_date': sf_date, 'buyer_inn':inn_buyer, 'buyer_kpp':kpp_buyer, 'seller_inn':inn_seller, 'seller_kpp':kpp_seller}

  def extract_sf_data(self,text):
    text = text.upper()    
    if self.is_no_sf(text):
      raise NoSfException()

    # делим по переводу каретки удаляем пустые строки
    text = [x for x in text.split('\n') if x.strip()]
    text = self.clean_text(text)
    sf_no, sf_date = self.extract_sf_no_and_date(text)

    inn_seller, kpp_seller, inn_buyer, kpp_buyer = self.extract_inn_kpp(text)

    return {'sf_no':sf_no, 'sf_date': sf_date, 'buyer_inn':inn_buyer, 'buyer_kpp':kpp_buyer, 'seller_inn':inn_seller, 'seller_kpp':kpp_seller}

  def correct_sf_num(self, t):
    # выкидывает не цифры и буквы с концов строки
    l = 0
    for x in t:
      if x.isdigit() or x.isalpha():
        break
      l += 1

    r = 0
    for x in t[::-1]:
      if x.isdigit() or x.isalpha():
        break
      r += 1

    return t[l:len(t)-r]
  
  @logger_wraps()
  def extract_sf_no(self,sf_no):
    if not sf_no.strip():
      return ''

    sf_no = self.config.PATTERN_REPLACE.sub('', sf_no)

    match = re.findall(r'\d+[А-Я]+|\d+', sf_no)
    if not match:
      return ''
    match = match[-1]
    sf_no = sf_no[:sf_no.rfind(match)+len(match)]
    sf_no = ''.join([x for x in sf_no if x not in ' ' ])

    return self.correct_sf_num(sf_no)  

  def find_inn_kpp_depth(self, texts,i,depth=2,direction='top-bottom'):
    top = texts[max(0,i-depth):i][::-1]
    bottom = texts[i+1:min(len(texts),i+depth+1)]
    # ищем вверх на depth потом ищем вниз на depth
    search_list = [top,bottom]  
    # иначе наоборот
    if direction!='top-bottom':
      search_list = search_list[::-1]
    # logger.debug(search_list)
    for lst in search_list:    
      for text in lst:
        if len(text)>25:
          continue
        text =  clean_space(text)
        inn_kpp_data = self.config.PATTERN_INN_KPP.findall(text)    
        if inn_kpp_data:
          return inn_kpp_data[0]
    return None, None

  def extract_inn_kpp(self, text):
    inn_s, kpp_s, inn_b, kpp_b = None,None,None,None

    s,b = 0,0
    groups,groupb = '',''  
    try:
      s,groups = group_by_pattern(self.config.PATTERN_INN_KPP_SELLER,text)
      if len(groups)<19:
        # может быть второе вхождение
        _,groups = group_by_pattern(self.config.PATTERN_INN_KPP_SELLER,text[s+1:])
        text.pop(s)
    except:
      pass
    try:  
      b,groupb = group_by_pattern(self.config.PATTERN_INN_KPP_BUYER,text)      
      if len(groupb)<19:
        # может быть второе вхождение
        _,groupb = group_by_pattern(self.config.PATTERN_INN_KPP_BUYER,text[b+1:])
        text.pop(b)
    except Exception as e:
      pass

    logger.debug(('PATTERN_SELLER',groups))
    logger.debug(('PATTERN_BUYER',groupb))
    groups = clean_space(groups)
    inn_kpp_data_s = self.config.PATTERN_INN_KPP.findall(groups)
    if inn_kpp_data_s:
      inn_s, kpp_s = inn_kpp_data_s[0]
    else:
      direction = 'top-bottom' if s<b else 'bottom-top'
      inn_s, kpp_s = self.find_inn_kpp_depth(text,s,direction=direction)

    groupb = clean_space(groupb)
    inn_kpp_data_b = self.config.PATTERN_INN_KPP.findall(groupb)
    if inn_kpp_data_b:
      inn_b, kpp_b = inn_kpp_data_b[0]
    else:
      direction = 'bottom-top' if s<b else 'top-bottom'
      inn_b, kpp_b = self.find_inn_kpp_depth(text,b,direction=direction)
        
    
    return inn_s, kpp_s, inn_b, kpp_b

  def extract_inn_kpp1(self, pattern, text):
    inn, kpp = None,None
    try:
      i,group = group_by_pattern(pattern,text)

      inn_kpp_data = self.config.PATTERN_INN_KPP.findall(group)    
      if not inn_kpp_data:
        group = text[i+1]
        inn_kpp_data = self.config.PATTERN_INN_KPP.findall(group)    
        if not inn_kpp_data:
          group = text[i-1]
          inn_kpp_data = self.config.PATTERN_INN_KPP.findall(group)    
      if inn_kpp_data:
        inn, kpp = inn_kpp_data[0]
    except:
      pass
    return inn, kpp

  def is_no_sf(self, text):
    # документ не с/ф?
    return not self.config.PATTERN_NOT_SF.search(text) is None

  def  clean_text(self, text):    
    # фильтруем хлам вида ^(*)$    
    text = list(filter(lambda x: (not self.config.TRASH_PATTERN.search(x)), text))
    text = list(filter(lambda x: (not self.config.PATTERN_TRASH_WORD.search(x)) | (not self.config.PATTERN_BILL.search(x) is None), text))      

    # удаляем ненужные символы
    for i,t in enumerate(text):
      t = ''.join([x for x in t if x not in '_—,\[\]`‘\'"~!@#$%^&*();:?*+=|\\'])
      text[i] = t

    # удалить пустые строки заменить сдвоенные пробелы
    text = [re.sub(' +',' ',x) for x in text if len(x.strip())>1]
    text = [re.sub('\. ','.',x) for x in text]

    return text

  def is_eng_sf_num(self, info):
    if info['seller_inn'] == '6665002150' and info['seller_kpp'] == '660850001':
      return True
    return False

  def find_sf_num(self, data, left_break):
    text, coords = format_text(data, left_break)
    logger.debug(text)
    i,group = group_by_pattern(self.config.PATTERN_SF_NUM,text)
    if len(group)>10:
      group = group.strip()    
      left_part_text = group.split(' ')[0]
      
      try:
        idx = find_part_text(data, left_part_text)      
        left = data['left'][idx]
        _,top,width,height = coords[i]
        return [(group,(left,top,width,height))]
      except: 
        left,top,width,height = coords[i]
        return [(group,(left,top,width,height))]

    left, top, width, height = coords[i]
    idx = [i for i, x in enumerate(coords) if len(text[i]) < 60 and left+width<x[0] and top < x[1]+(x[3]/2) < top + height ]
    
    return [(text[i],coords[i]) for i in idx]


  def correct_eng_sf_num(self, t):  
    sf_no = self.config.PATTERN_ENG_DATE_SPLIT.split(t.upper())[0]
    sf_no = ''.join([x for x in sf_no if x not in ' ' ])
    sf_no = sf_no.replace('-KJ','-KI').replace('-K1I','-KI').replace('-KI1','-KI').replace('-K1','-KI')
    return self.correct_sf_num(sf_no)  

  def extract_eng_sf_num(self,head_img):
    config = '--oem 1 --psm 11'
    data = ocr_data(head_img,lang='rus',config=config)

    _,w = head_img.shape[:2]
    left_break = w // 2
    info = self.find_sf_num(data,left_break)[0]
    _,(x,y,w,h) = info
    x -= 4
    y -= 4
    w += 8
    h += 8

    box = head_img[y:y+h, x:x+w]

    config = '--oem 1 --psm 7'
    sf_num = ocr(box,lang='eng',config=config)

    return self.correct_eng_sf_num(sf_num)        
  
  def process(self, img):    
    # text = self.text_from_img(img)    
    h,w = img.shape[:2]
    d = dpi(w,h)
    # заголовок страницы с инфо по с/ф
    head_img = self.head_roi(img)
    # head_img = increase_brightness(head_img,15)
    # ocr
    text = ocr_rus(head_img, config = f'--oem 1 --psm 11 --dpi {d}')    

    try:      
      info = self.extract_sf_data(text)      

      if self.is_eng_sf_num(info):
        try:
          info['sf_no'] = self.extract_eng_sf_num(head_img)  
        except Exception as e:
          logger.exception(e)

      # если встретилось не заполненное поле
      for k in info.keys():
        if not info[k]:          
          # провести ocr --psm 6 и заполнить не проставленные поля
          text1 = ocr_rus(head_img, config = f'--oem 1 --psm 6')
          info1 = self.extract_sf_data(text1) 
          for k1 in info.keys():
            if bool(not info[k1]) & bool(info1[k1]):
              info[k1] = info1[k1]
          text += '\n++++++++++++++++++++\n' + text1
          break

      dtext = '\n'.join([ x for x in text.split('\n') if len(x)>2 ])
      logger.debug(dtext)
      logger.debug(info)
      return info, text
    except NoSfException as ex:
      logger.error('no sf')
      return {'sf_no':None, 'sf_date': None, 'buyer_inn':None, 'buyer_kpp':None, 'seller_inn':None, 'seller_kpp':None}, None
    except Exception as e:
      logger.exception(e)

  def head_roi1(self, img):
    gray =preprocess_image(img, method=4, kernel_size=3)
    # gray = to_gray(img)
    bw = to_binary(gray)
    hor_min_size = bw.shape[1] // 20
    ver_min_size =  bw.shape[0] // 15 # 200
    horizontal = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (hor_min_size, 1)) )
    vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )  

    mask = horizontal+vertical  
    x,y,w,h = table_roi(mask) 
    logger.debug('table roi {}',(x,y,w,h)) 
    
    mask = mask[y:y+h, x:x+w]  
    # mask = horizontal[y:y+h, x:x+w]  
    angle = calculate_angle(horizontal)
    if y < mask.shape[0]//10:
      y = mask.shape[0] // 4

    img = img[0:y,0:img.shape[1]] 
    _,img = rotation(img,angle)  

    # отрезать от вертикальной черты если есть
    x,y,w,h = self.right_region(img)
    roi = img[y:y+h, x:x+w]  

    # roi = remove_ticket(roi)

    return roi 

  def head_roi(self, img):
    gray =preprocess_image(img, method=4, kernel_size=3)
    bw = to_binary(gray)

    hor_min_size = bw.shape[1] // 10
    ver_min_size =  bw.shape[0] // 10 # 200
    horizontal = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (hor_min_size, 1)) )
    vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )  

    min_y,max_y = table_roi(horizontal,vertical)     
    logger.debug('table roi {}',(min_y,max_y)) 
    y = min_y

    # x,y,w,h = table_roi1(horizontal+vertical)    
    # logger.debug('table roi {}',(x,y,w,h))  
    

    if y < img.shape[0]//4:      
      y = int(img.shape[0] // 2.5)

    angle = calculate_angle(horizontal)
    
    img = img[0:y,0:img.shape[1]] 
    _,img = rotation(img,angle)  
    

    # отрезать от вертикальной черты если есть
    x,y,w,h = self.right_region(img)
    roi = img[y:y+h, x:x+w]  

    if roi.shape[1]==0:      
      return img

    # roi = remove_ticket(roi)

    return roi 


  def head_roi2(self, img):
    gray =preprocess_image(img, method=4, kernel_size=3)
    # gray = to_gray(img)
    bw = to_binary(gray)
    hor_min_size = bw.shape[1] // 10
    ver_min_size =  bw.shape[0] // 10 # 200
    horizontal = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (hor_min_size, 1)) )
    vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )  

    mask = horizontal+vertical  
    # x,y,w,h = table_roi1(horizontal,vertical) 
    # logger.debug('table roi {}',(min_y,max_y)) 
    min_y,max_y = table_roi(horizontal,vertical)     
    y = min_y
    h = max_y-y
    x = 0
    w = mask.shape[1]

    mask = mask[y:y+h, x:x+w]

    # mask = horizontal[y:y+h, x:x+w]  
    angle = calculate_angle(horizontal)
    
    if y < mask.shape[0]//4:      
      y = int(mask.shape[0] // 2.5)

    img = img[0:y,0:img.shape[1]] 
    _,img = rotation(img,angle)  

    # отрезать от вертикальной черты если есть
    x,y,w,h = self.right_region(img)
    roi = img[y:y+h, x:x+w]  

    # roi = remove_ticket(roi)

    return roi 

  def right_region(self, gray):
    ''' отрезать ROI от вертикальной черты если есть '''
    gray =preprocess_image(gray, method=4, kernel_size=3)
    bw = to_binary(gray)
    ver_min_size =  bw.shape[0] // 10 # 200    
    vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )

    x,y,w,h = 0,0,bw.shape[1],bw.shape[1]
    vert_contours,v_min_x,v_min_y,v_max_x,v_max_y = get_contours(vertical)

    if not vert_contours:
      return x,y,w,h

    contour_list = [(cnt, cv2.contourArea(cnt)) for cnt in vert_contours]
    # (n,(x,y,w,h))
    vert_rect_list = [(i,cv2.boundingRect(x[0])) for i,x in enumerate(contour_list) ]

    # max h contour
    max_h_contour_idx = np.argmax([x[1][3] for x in vert_rect_list ])

    x1,y1,w1,h1 = vert_rect_list[max_h_contour_idx][1]
    if h1 > bw.shape[0] / 2 and x1 < bw.shape[1] / 2 :
      return x1+w1,0,bw.shape[1]-x1,bw.shape[0]
      # return x1+w1,max(0,y1-5),bw.shape[1]-x1,y1+h1
    return x,y,w,h


  def text_from_img(self, img):
    h,w = img.shape[:2]
    d = dpi(w,h)
    # заголовок страницы с инфо по с/ф
    head_img = self.head_roi(img)
    # cv2_imshow(head_img)

    # ocr
    return ocr_rus(head_img, config = '--oem 1 --psm 6')
    # return ocr_rus(head_img, config = f'--oem 1 --psm 11 --dpi {d}')
