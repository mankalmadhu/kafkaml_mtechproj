ARG BASEIMG=python:3.9
FROM --platform=linux/arm64 ${BASEIMG}

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librdkafka-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/app/requirements.txt

# Install scikit-learn separately with specific build configuration

# Install remaining requirements
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app/

RUN chmod +x ./start.sh
CMD ["./start.sh"]