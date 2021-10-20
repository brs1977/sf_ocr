#!/bin/bash

#/home/ruslan/prj/sf_ocr/venv/bin/uvicorn server:app --reload --host 0.0.0.0 --port 9091 --workers 4
/home/ruslan/prj/sf_ocr/venv/bin/uvicorn server2:app --host 0.0.0.0 --port 9091 --workers 1 --app-dir .

# app="www_ocr"

# # docker build -t ${app} .

# docker run -d -p 9095:9095 ${app} --name ${app} --rm