SHELL := /bin/bash

venv-init:
	pip install virtualenv

	python -m venv venv
	source venv/bin/activate

	pip install --upgrade pip
	pip install wheel

	pip install -r requirements.txt

gdown-models:
	mkdir -p models
	gdown --id 1-Q5BGBKs53ZsZZXgzTnNVxKq78ScZa20 -O models/type.pkl
	gdown --id 1-PlVu3-wGVVIiBb2fcEpHzr-SLLSrw-x -O models/orient.pkl
	cp config.yaml models

tesseract-install:
	apt-get install -y python3-pip python3-dev build-essential libgl1-mesa-dev curl autoconf libtool libleptonica-dev
	wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.0.0-rc1.zip
	unzip 5.0.0-rc1.zip

	cd tesseract-5.0.0-rc1
	autoreconf --install
	./configure
	make
	make install
	ldconfig

	wget https://github.com/tesseract-ocr/tessdata/blob/main/rus.traineddata?raw=true -O /usr/local/share/tessdata/rus.traineddata

dataset-load:
