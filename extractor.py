from img_utils import *
import functools
from loguru import logger

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

def group_by_pattern(pattern,text):
  for i,t in enumerate(text):
    grp = pattern.findall(text[i].upper())
    if grp:
      return i, grp[0][1]
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
    
    split_data = self.config.PATTERN_DATE_SPLIT.split(dt)

    if len(split_data)==1 and len(split_data[0])>10:
      return ''
    elif len(split_data)==3:
      split_data = [x.strip() for x in split_data]
      d,m,y = split_data
      m = self.config.MONTH[m]   
      return to_date(d,m,y)

    dt = dt.replace('-','.')
    dt = dt.replace('/','.')

    dt_split = dt.split('.')
    if len(dt_split) == 3:
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


  def extract_sf_data(self,text):
    sf_no, sf_date = None, None
    try:
      try:
        #ищем по патерну с/ф
        i,sf_no_group = group_by_pattern(self.config.PATTERN_SF_NUM,text)  
      except:
        # если счет/фактура разбита на 2 слова, ищем по патерну счет или фактура
        i,sf_no_group = group_by_pattern(self.config.PATTERN_BILL,text)  
        sf_no_group = ' '.join([t for t in text[i:i+4] if len(t)<25])
        i,sf_no_group = group_by_pattern(self.config.PATTERN_SF_NUM,[sf_no_group])  

      # если пусто то данные по с/ф ниже 
      if not sf_no_group.strip():
        sf_no_group = ' '.join([t for t in text[i+1:i+4] if len(t)<25])

      # очистка от ошибок распознавания
      sf_no_group = ''.join([x for x in sf_no_group if x not in '`‘\'"~!@#$%^&*();:?*+=|\\'])
      sf_no_group = sf_no_group.upper()
      
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
        sf_date = self.extract_date(sf_date.upper())
    except:
      pass

    inn_seller, kpp_seller = self.extract_inn_kpp(self.config.PATTERN_SELLER, text)
    inn_buyer, kpp_buyer = self.extract_inn_kpp(self.config.PATTERN_BUYER, text)

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


  def extract_inn_kpp(self, pattern, text):
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

  def process(self, img):    
    try:
      text = self.text_from_img(img)    
      text = [x for x in text.split('\n') if x.strip()]
      
      info = self.extract_sf_data(text)        
      logger.debug('\n'.join(text))
      logger.debug(info)
      return info
    except Exception as e:
      logger.error(e)

  def head_roi(self, img):
    gray =preprocess_image(img, method=4, kernel_size=3)
    # gray = to_gray(img)
    bw = to_binary(gray)
    hor_min_size = bw.shape[1] // 20
    ver_min_size =  bw.shape[0] // 15 # 200
    horizontal = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (hor_min_size, 1)) )
    vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )  

    mask = horizontal+vertical  
    x,y,w,h = table_roi(mask)  
    
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
      return x1+w1,max(0,y1-5),bw.shape[1]-x1,y1+h1
    return x,y,w,h


  def text_from_img(self, img):
    # заголовок страницы с инфо по с/ф
    head_img = self.head_roi(img)
    # cv2_imshow(head_img)

    # ocr
    return ocr_rus(head_img, config='--oem 1 -- psm 11')

# class SfInfoExtractor:
#   def __init__(self, config):
#     self.config = config

#   @logger_wraps()
#   def extract_date(self, dt):  
#     def to_date(d,m,y):
#       for x in self.config.NUMS_LETTER_CORRECT:
#         d = d.replace(x,str(self.config.NUMS_LETTER_CORRECT[x]))

#       m = f'{int(m):02d}'
#       d = f'{int(d):02d}'
#       return '.'.join([d,m,y])

#     if not dt:
#       return None

#     split_data = self.config.PATTERN_DATE_SPLIT.split(dt)
#     if len(split_data)==3:
#       split_data = [x.strip() for x in split_data]
#       d,m,y = split_data
#       m = self.config.MONTH[m]   
#       return to_date(d,m,y)

#     dt = dt.replace('-','.')
#     dt = dt.replace('/','.')

#     dt_split = dt.split('.')
#     if len(dt_split) == 3:
#       d, m, y = dt_split
#     elif len(dt_split) == 2:
#       d, m = dt_split
#       if len(m)<=4:        
#         y = m[-2:]
#         m = m[:-2]
#       else:
#         y = m[4:]
#         m = m[:-4]
#     return to_date(d,m,y)


#   def extract_sf_data(self, sf_data):
#     if not sf_data:
#       return None, None

#     sf_data = ''.join([x for x in sf_data if x not in '`‘\'"~!@#$%^&*();:?*+=|\\'])

#     sf_no, sf_date = None, None 
#     res = self.config.PATTERN_DATE_SEARCH.search(sf_data.upper())
#     if res:
#       sf_no, sf_date = res.groups()

#     sf_date = self.extract_date(sf_date)

#     if not sf_no:
#       return sf_no, sf_date

#     # оставляем цифры и -
#     # sf_no = ''.join([x for x in sf_no if x.isdigit() or x in '-' ])

#     sf_no = self.config.PATTERN_REPLACE.sub('', sf_no)

#     match = re.findall(r'\d+[А-Я]+|\d+', sf_no)[-1]
#     sf_no = sf_no[:sf_no.rfind(match)+len(match)]
#     sf_no = ''.join([x for x in sf_no if x not in ' ' ])
    
#     sf_no = self.correct_sf_num(sf_no)
#     return sf_no, sf_date

#   def correct_sf_num(self, t):
#     # выкидывает не цифры и буквы с концов строки
#     l = 0
#     for x in t:
#       if x.isdigit() or x.isalpha():
#         break
#       l += 1

#     r = 0
#     for x in t[::-1]:
#       if x.isdigit() or x.isalpha():
#         break
#       r += 1

#     return t[l:len(t)-r]


#   @logger_wraps()
#   def extract_inn_kpp_data(self, inn_kpp_data):
#     if not inn_kpp_data:
#       return None, None

#     inn_kpp_data = self.config.PATTERN_INN_KPP.findall(inn_kpp_data)    
#     if not inn_kpp_data:
#       return None, None
#     return inn_kpp_data[0]

#   def process(self, img):    
#     text = self.text_from_img(img)    
#     text = text.split('\n')
#     info = self.sf_info_from_img_text(text)        
#     logger.debug('\n'.join([x for x in text if x.strip()]))
#     logger.debug(info)
#     return info

#   def sf_info_from_img_text(self, arr):
#     sf_no = regexp_group_by_pattern(arr,self.config.PATTERN_SF_NUM)
#     b_inn_kpp = regexp_group_by_pattern(arr,self.config.PATTERN_BUYER)
#     s_inn_kpp = regexp_group_by_pattern(arr,self.config.PATTERN_SELLER)
#     if b_inn_kpp:
#       b_inn_kpp = ''.join([x for x in b_inn_kpp if x!=' '])
#     if s_inn_kpp:
#       s_inn_kpp = ''.join([x for x in s_inn_kpp if x!=' '])


#     sf_no, sf_date = self.extract_sf_data(sf_no)
#     b_inn, b_kpp = self.extract_inn_kpp_data(b_inn_kpp)
#     s_inn, s_kpp = self.extract_inn_kpp_data(s_inn_kpp)

#     return {'sf_no':sf_no, 'sf_date': sf_date, 'buyer_inn':b_inn, 'buyer_kpp':b_kpp, 'seller_inn':s_inn, 'seller_kpp':s_kpp}

#   def cut_right_side_img(self, img):
#     ''' отрезать ROI от вертикальной черты если есть '''
#     gray =preprocess_image(img, method=4, kernel_size=3)
#     bw = to_binary(gray)
#     ver_min_size =  bw.shape[0] // 10 # 200
#     vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )


#     vert_contours,v_min_x,v_min_y,v_max_x,v_max_y = get_contours(vertical)

#     if not vert_contours:
#       return img

#     contour_list = [(cnt, cv2.contourArea(cnt)) for cnt in vert_contours]
#     # (n,(x,y,w,h))
#     vert_rect_list = [(i,cv2.boundingRect(x[0])) for i,x in enumerate(contour_list) ]

#     # max h contour
#     max_h_contour_idx = np.argmax([x[1][3] for x in vert_rect_list ])

#     x,y,w,h = vert_rect_list[max_h_contour_idx][1]
#     # print(x, img.shape[1] / 3)
#     if h > img.shape[0] / 2 and x < img.shape[1] / 2 :
#       img=img[y:y+h, x+w:img.shape[1]-x]
#       # cv2_imshow(img)

#     return img

#   def get_head_img(self, img):
#     gray =preprocess_image(img, method=4, kernel_size=3)
#     bw = to_binary(gray)
#     hor_min_size = bw.shape[1] // 20
#     ver_min_size =  bw.shape[0] // 15 
#     horizontal = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (hor_min_size, 1)) )
#     vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )

#     mask = horizontal+vertical
#     x,y,w,h = table_roi(mask)  
#     roi = mask[y:y+h, x:x+w]
#     angle = calculate_angle(roi)
#     # rm,img = rotation(img,angle)

#     # убираем гор линии по маске
#     gray = to_gray(img)
#     mask = cv2.dilate(mask, (3,3), iterations=3)
#     gray = cv2.addWeighted(gray, 1, mask, 1, 0.0)
#     rm,img = rotation(gray,angle)  


#     if y < img.shape[0]//10:
#       y = img.shape[0] // 4
#     return img[0:y,0:img.shape[1]]



#   def text_from_img(self, img):
#       # заголовок страницы с инфо по с/ф
#       head_img = self.get_head_img(img)
#       # cv2_imshow(head_img)

#       # отрезать от вертикальной черты если есть
#       head_img = self.cut_right_side_img(head_img)

#       # cv2_imshow(head_img)

#       # size = tuple(map(lambda x : x // 3, head_img.shape[:2][::-1]))    
#       # cropped_img = head_img.copy()
#       # cropped_img = cv2.resize(head_img, size )
#       # cv2_imshow(cropped_img)
#       # ocr
#       return ocr_rus(head_img)

# extractor = SfInfoExtractor(config)