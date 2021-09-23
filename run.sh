#!/bin/bash
# uvicorn server:app --reload --host 0.0.0.0 --port 9095

app="www_ocr"

# docker build -t ${app} .

docker run -d -p 9095:9095 ${app} 