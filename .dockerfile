FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
WORKDIR /app
COPY . /app
RUN pip install pandas numpy Flask flask-mysql
CMD ["python3", "server.py"]
#CMD ['server.py']