ARG BASEIMG=kafkaml-base:latest
FROM ${BASEIMG}


# set work directory
WORKDIR /usr/src/app

# set environment variables
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app/

RUN chmod +x ./start.sh
CMD ["./start.sh"]
