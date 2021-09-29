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
    if len(split_data)==3:
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


  def extract_sf_data(self, sf_data):
    if not sf_data:
      return None, None

    sf_data = ''.join([x for x in sf_data if x not in '`‘\'"~!@#$%^&*();:?*+=|\\'])

    sf_no, sf_date = None, None 
    res = self.config.PATTERN_DATE_SEARCH.search(sf_data.upper())
    if res:
      sf_no, sf_date = res.groups()

    sf_date = self.extract_date(sf_date)

    if not sf_no:
      return sf_no, sf_date

    # оставляем цифры и -
    # sf_no = ''.join([x for x in sf_no if x.isdigit() or x in '-' ])

    sf_no = self.config.PATTERN_REPLACE.sub('', sf_no)

    match = re.findall(r'\d+[А-Я]+|\d+', sf_no)[-1]
    sf_no = sf_no[:sf_no.rfind(match)+len(match)]
    sf_no = ''.join([x for x in sf_no if x not in ' ' ])
    
    sf_no = self.correct_sf_num(sf_no)
    return sf_no, sf_date

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
  def extract_inn_kpp_data(self, inn_kpp_data):
    if not inn_kpp_data:
      return None, None

    inn_kpp_data = self.config.PATTERN_INN_KPP.findall(inn_kpp_data)    
    if not inn_kpp_data:
      return None, None
    return inn_kpp_data[0]

  def process(self, img):    
    text = self.text_from_img(img)    
    text = text.split('\n')
    info = self.sf_info_from_img_text(text)        
    logger.debug('\n'.join([x for x in text if x.strip()]))
    logger.debug(info)
    return info

  def sf_info_from_img_text(self, arr):
    sf_no = regexp_group_by_pattern(arr,self.config.PATTERN_SF_NUM)
    b_inn_kpp = regexp_group_by_pattern(arr,self.config.PATTERN_BUYER)
    s_inn_kpp = regexp_group_by_pattern(arr,self.config.PATTERN_SELLER)
    if b_inn_kpp:
      b_inn_kpp = ''.join([x for x in b_inn_kpp if x!=' '])
    if s_inn_kpp:
      s_inn_kpp = ''.join([x for x in s_inn_kpp if x!=' '])


    sf_no, sf_date = self.extract_sf_data(sf_no)
    b_inn, b_kpp = self.extract_inn_kpp_data(b_inn_kpp)
    s_inn, s_kpp = self.extract_inn_kpp_data(s_inn_kpp)

    return {'sf_no':sf_no, 'sf_date': sf_date, 'buyer_inn':b_inn, 'buyer_kpp':b_kpp, 'seller_inn':s_inn, 'seller_kpp':s_kpp}

  def cut_right_side_img(self, img):
    ''' отрезать ROI от вертикальной черты если есть '''
    gray =preprocess_image(img, method=4, kernel_size=3)
    bw = to_binary(gray)
    ver_min_size =  bw.shape[0] // 10 # 200
    vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )


    vert_contours,v_min_x,v_min_y,v_max_x,v_max_y = get_contours(vertical)

    if not vert_contours:
      return img

    contour_list = [(cnt, cv2.contourArea(cnt)) for cnt in vert_contours]
    # (n,(x,y,w,h))
    vert_rect_list = [(i,cv2.boundingRect(x[0])) for i,x in enumerate(contour_list) ]

    # max h contour
    max_h_contour_idx = np.argmax([x[1][3] for x in vert_rect_list ])

    x,y,w,h = vert_rect_list[max_h_contour_idx][1]
    # print(x, img.shape[1] / 3)
    if h > img.shape[0] / 2 and x < img.shape[1] / 2 :
      img=img[y:y+h, x+w:img.shape[1]-x]
      # cv2_imshow(img)

    return img

  def get_head_img(self, img):
    gray =preprocess_image(img, method=4, kernel_size=3)
    bw = to_binary(gray)
    hor_min_size = bw.shape[1] // 20
    ver_min_size =  bw.shape[0] // 15 
    horizontal = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (hor_min_size, 1)) )
    vertical   = to_lines(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_size)) )

    mask = horizontal+vertical
    x,y,w,h = table_roi(mask)  
    roi = mask[y:y+h, x:x+w]
    angle = calculate_angle(mask)
    rm,img = rotation(img,angle)  
    if y < img.shape[0]//10:
      y = img.shape[0] // 4
    return img[0:y,0:img.shape[1]]



  def text_from_img(self, img):
      # заголовок страницы с инфо по с/ф
      head_img = self.get_head_img(img)
      # cv2_imshow(head_img)

      # отрезать от вертикальной черты если есть
      head_img = self.cut_right_side_img(head_img)

      # cv2_imshow(head_img)

      # size = tuple(map(lambda x : x // 3, head_img.shape[:2][::-1]))    
      # cropped_img = head_img.copy()
      # cropped_img = cv2.resize(head_img, size )
      # cv2_imshow(cropped_img)
      # ocr
      return ocr_rus(head_img)

# extractor = SfInfoExtractor(config)