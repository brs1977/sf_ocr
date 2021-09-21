FROM python:3.8-slim-buster
#FROM python:3.9.7-slim-buster

ARG USERNAME=alex
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
        && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get install -y --no-install-recommends \
    tesseract-ocr-rus \
    python3-pip \
    python3-opencv \
	binutils libc6 \
	binutils libc-bin \
    python-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt /tmp
RUN pip3 install wheel
RUN python -m pip install -r /tmp/requirements.txt

WORKDIR /usr/src/app

COPY *.py /usr/src/app/
RUN pyinstaller ocr.py --onefile 

RUN mkdir dist/models 
COPY config.yaml dist/models
#COPY models/*.pkl dist/models/

RUN gdown --id 1-172My1T8VvCSPCc4eHcJZXkZvuXlo9V -O dist/models/type.pkl \
    && gdown --id 10scZJVWomVktMUdjd1QIlwxd1kXclmwT -O dist/models/orient.pkl

USER $USERNAME 
