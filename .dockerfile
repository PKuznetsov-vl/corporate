FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
WORKDIR /app
COPY . /app
RUN pip pandas numpy Flask flask-mysql
ENTRYPOINT ['python']
CMD ['server.py']