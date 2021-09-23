FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
RUN apt-get update -y   
RUN apt-get install -y python-pip python-dev build-essential libgl1-mesa-dev tesseract-ocr-rus

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt 

RUN mkdir /usr/src/app/output
COPY models /usr/src/app
COPY *.py /usr/src/app/
WORKDIR /usr/src/app

# ENTRYPOINT ["unicorn"]   
CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "9090"]