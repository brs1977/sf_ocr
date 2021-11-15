import yaml
import re

class Config:
  def __init__(self, file_name):
    with open(file_name, "r") as f:
      data = yaml.safe_load(f)    

    self.PATTERN_BILL = re.compile(data['bill_template'])    
    self.PATTERN_NOT_SF = re.compile('|'.join(data['not_sf_template']))

    self.PATTERN_SF_NUM = re.compile(r'.*?('+'|'.join(data['sf_num_template'])+')(.*)')
    self.PATTERN_SELLER = re.compile(f".*?([{'|'.join(data['inn_kpp_template'])}] {data['seller_template']})(.*)")
    self.PATTERN_BUYER  = re.compile(f".*?([{'|'.join(data['inn_kpp_template'])}] {data['buyer_template']})(.*)")
    
    self.MONTH = data['month']
    
    self.PATTERN_MONTH = '|'.join([x for x in self.MONTH])

    # встретиться буква З В
    PATTERN_DAY = '[\d'+''.join([x for x in data['nums_letter_correct']])+']{1,2}'  
    self.PATTERN_DATE = re.compile('('+PATTERN_DAY+'|\d).(\d{1,2}).?(\d{4}|\d{2})')
    self.PATTERN_DATE_SEARCH = re.compile(r'(.*?)('+PATTERN_DAY+'[\.|-]\d{1,2}[\.|-](?:\d{4}|\d{2})|'+PATTERN_DAY+' ?(?:'+self.PATTERN_MONTH+') ?(?:\d{4}|\d{2}))')
    self.PATTERN_INN_KPP = re.compile(data['inn_kpp_num_template'])
    self.PATTERN_DATE_SPLIT = re.compile(r'('+self.PATTERN_MONTH+')') 
    self.NUMS_LETTER_CORRECT = data['nums_letter_correct']
    self.PATTERN_REPLACE = re.compile('|'.join([x for x in data['replace_template']]))
    self.TRASH_PATTERN = re.compile(data['trash_template'])

    self.PATTERN_TRASH_WORD = re.compile('|'.join(data['trash_word_template']))
    
    