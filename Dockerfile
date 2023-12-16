# Download base image ubuntu 22.04
FROM ubuntu:18.04

# LABEL about the custom image
LABEL maintainer="jcsousa@cpqd.com.br"
LABEL version="0.1"
LABEL description="This is a custom Docker Image for Generative Adversarial Network to Generate Images."

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
COPY GAN.py /home/
COPY data/* /home/data/
RUN apt update -y && apt upgrade -y && apt install vim -y && apt install git -y
RUN cd /home/ && chmod +x GAN.py
RUN apt install python3 -y && apt install python3-pip -y && pip3 install "protobuf>=3.11.0, <=3.20.1" && pip3 install tensorflow && pip3 install numpy && pip3 install keras && pip3 install --upgrade pip && pip install opencv-python && pip install progressbar2 && pip install scipy==1.1.0 && pip install pillow && pip install matplotlib 

CMD python3 ./GAN.py

WORKDIR /home
