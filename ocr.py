from hog_classifier import HogClassifier, load_model
from config import Config
from pdf_utils import * 
from extractor import SfInfoExtractor

import argparse
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', action='store', dest='pdf_file', help='Input pdf file', required=True)
    parser.add_argument('-out', action='store', dest='output_path', help='Output path', default="output")

    args  = parser.parse_args()

    # print(args .pdf_file, args.output_path)

    config = Config('models/config.yaml')
    extractor = SfInfoExtractor(config)
    orient_clf = load_model('models/orient.pkl')
    type_clf = load_model('models/type.pkl')

    split_pdf(args.pdf_file,orient_clf,type_clf,extractor,path=args.output_path)
