#!/bin/bash

sudo apt-get install tesseract-ocr-rus
sudo apt-get install python3-pip
sudo apt-get update
sudo apt-get install -y python3-opencv
sudo apt-get install python-dev
sudo apt-get install python3-dev

pip3 install virtualenv

python3 -m venv venv
source venv/bin/activate

pip3 install --upgrade pip
pip3 install wheel

pip3 install -r requirements.txt


#compile
# cp ocr.py ocr.pyx
# cython ocr.pyx --embed
# gcc -Os -I /usr/include/python3.6 -o ocr ocr.c -lpython3.6 -lpthread -lm -lutil -ldl

pip3 install pyinstaller
pyinstaller ocr.py --onefile


mkdir dist/models

cp config.yaml dist/models
gdown --id 1-172My1T8VvCSPCc4eHcJZXkZvuXlo9V -O dist/models/type.pkl
gdown --id 10scZJVWomVktMUdjd1QIlwxd1kXclmwT -O dist/models/orient.pkl
