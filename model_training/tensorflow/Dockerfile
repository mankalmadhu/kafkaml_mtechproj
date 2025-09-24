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
    wget \
    unzip \
    curl \
    git \
    cmake \
    ninja-build \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt


ARG SOLIDITY_VERSION=0.8.20

# Build Solidity compiler from source for ARM64
WORKDIR /tmp
RUN wget https://github.com/ethereum/solidity/releases/download/v${SOLIDITY_VERSION}/solidity_${SOLIDITY_VERSION}.tar.gz
RUN tar -xzf solidity_${SOLIDITY_VERSION}.tar.gz
WORKDIR /tmp/solidity_${SOLIDITY_VERSION}
RUN mkdir build
WORKDIR /tmp/solidity_${SOLIDITY_VERSION}/build
RUN cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DTESTS=OFF
RUN ninja
RUN cp solc/solc /usr/local/bin/solc
RUN chmod +x /usr/local/bin/solc

# Create the directory structure that py-solc-x expects
RUN mkdir -p /root/.solcx

# Copy the built solc to the expected location
RUN cp /usr/local/bin/solc /root/.solcx/solc-v${SOLIDITY_VERSION}

# Verify that the binary works
RUN solc --version
RUN /root/.solcx/solc-v${SOLIDITY_VERSION} --version

WORKDIR /usr/src/app
# Install remaining requirements
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV LD_PRELOAD=/usr/local/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# copy project
COPY . /usr/src/app/


RUN chmod +x ./start.sh
CMD ["./start.sh"]
