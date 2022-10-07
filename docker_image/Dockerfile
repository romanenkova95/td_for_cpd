FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
RUN apt-get update && apt-get install apt-file -y && apt-file update
RUN apt-get install -y git build-essential curl wget software-properties-common zip unzip vim
RUN pip install jupyterlab 
EXPOSE 5000
EXPOSE 9999
EXPOSE 10000

COPY requirements.txt /home/requirements.txt
WORKDIR /home/
RUN pip install -r requirements.txt
