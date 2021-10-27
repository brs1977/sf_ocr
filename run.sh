#!/bin/bash

#/home/ruslan/prj/sf_ocr/venv/bin/uvicorn server:app --reload --host 0.0.0.0 --port 9091 --workers 4
# /home/ruslan/prj/sf_ocr/venv/bin/uvicorn server:app --host 0.0.0.0 --port 9091 --workers 1 --app-dir .
/home/ruslan/prj/sf_ocr/venv/bin/uvicorn server:app --host 0.0.0.0 --port 9091 --workers 1 --app-dir .

# app="www_ocr"

# # docker build -t ${app} .

# docker run -d -p 9095:9095 ${app} --name ${app} --rm

# https://drive.google.com/file/d/1-Q5BGBKs53ZsZZXgzTnNVxKq78ScZa20/view?usp=sharing
# https://drive.google.com/file/d/1-PlVu3-wGVVIiBb2fcEpHzr-SLLSrw-x/view?usp=sharing