SHELL := /bin/bash

venv-init:
	pip install virtualenv

	python -m venv venv
	source venv/bin/activate

	pip install --upgrade pip
	pip install wheel

	pip install -r requirements.txt

models:
	mkdir -p models
	gdown --id 1-Q5BGBKs53ZsZZXgzTnNVxKq78ScZa20 -O models/type.pkl
	gdown --id 1-PlVu3-wGVVIiBb2fcEpHzr-SLLSrw-x -O models/orient.pkl
	cp config.yaml models

run:
	/home/ruslan/prj/sf_ocr/venv/bin/uvicorn server:app --host 0.0.0.0 --port 9091 --workers 1 --app-dir .

	# app="www_ocr"
	# # docker build -t ${app} .
	# docker run -d -p 9095:9095 ${app} --name ${app} --rm


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
	mkdir -p dataset
	mkdir -p dataset/png
	gdown --id 1yjUmW5N2Ato3YsYKbntpyLk4DZQyZreM
	unzip imgs.zip

	rm ./imgs/cDcOHtKTmDc5W4VuQCF0Hvg0eJiSd8wI74ASe8hJ_4_sf_0.png
	rm ./imgs/xAXL4pzcqoZxwp0LZO9cwMb4KYNzXJJrhKsKn1vu_1_sf_1.png
	rm ./imgs/yT8Ois52rv0GP5OqGwyjVkxAPaSq3E4cgfktFv9K_5_sf_0.png	
	rm ./imgs/2L7Tyset1siObeOZNnrbydkXnqXS0RJTNBRSQIkG_5_sf_0.png
	rm ./imgs/EfegFVcai3uNitkeuwwmZksVl5pImY775mXhSoN3_7_sf_0.png
	rm ./imgs/2L7Tyset1siObeOZNnrbydkXnqXS0RJTNBRSQIkG_7_sf_0.png 
	rm ./imgs/cDcOHtKTmDc5W4VuQCF0Hvg0eJiSd8wI74ASe8hJ_3_sf_0.png
	rm ./imgs/Gu4EKoSxskqej3JF1ycwnicWdH2QovGxgIdADBNs_8_sf_0.png 
	rm ./imgs/939wTY7ws3xNfDVmjyrewW2LwlCqzmpPenB8Rndo_1_sf_1.png

	mv ./imgs/svhEvdK9XsWb3iC1J4Otc0Nm9keMdbdOECRgLBe6_1_sf1_0.png ./imgs/svhEvdK9XsWb3iC1J4Otc0Nm9keMdbdOECRgLBe6_1_sf_1.png
	mv ./imgs/NVxLArRUgvcjr8oPTlL549urC2q4h6c0Rf6BWlLT_2_sf1_0.png ./imgs/NVxLArRUgvcjr8oPTlL549urC2q4h6c0Rf6BWlLT_2_sf_0.png
	mv ./imgs/davX4vAa0jJxAcK7eHayJ6Yve7DBxMul3e8K0289_8_sf_1.png ./imgs/davX4vAa0jJxAcK7eHayJ6Yve7DBxMul3e8K0289_8_sf1_0.png
	mv ./imgs/bS3vL3MvkpzuedS9xy69JLBm9NuD6tkk0su8qsws_1_sf_0.png ./imgs/bS3vL3MvkpzuedS9xy69JLBm9NuD6tkk0su8qsws_1_sf_1.png
	mv ./imgs/iizdgJzssY3Ic0Vs9zPbq9OoeCoo6Eb4lHvBj3Hz_9_sf_0.png ./imgs/iizdgJzssY3Ic0Vs9zPbq9OoeCoo6Eb4lHvBj3Hz_9_sf1_0.png
	mv ./imgs/DBISj5rdHUzmCN7R0Jz1UXZZUvYurXe58ldRRbks_23_sf_1.png ./imgs/DBISj5rdHUzmCN7R0Jz1UXZZUvYurXe58ldRRbks_23_sf_0.png

	mv ./imgs/*.png ./dataset/png
	rm -rf imgs

compile:
	# cp ocr.py ocr.pyx
	# cython ocr.pyx --embed
	# gcc -Os -I /usr/include/python3.6 -o ocr ocr.c -lpython3.6 -lpthread -lm -lutil -ldl

	pip3 install pyinstaller
	pyinstaller ocr.py --onefile


