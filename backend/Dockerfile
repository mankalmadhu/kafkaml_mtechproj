ARG BASEIMG=python:3.10

FROM --platform=linux/arm64 ${BASEIMG}

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies including build tools for Solidity
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librdkafka-dev \
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

# Install Foundry (Anvil) for local blockchain server
WORKDIR /tmp
RUN curl -L https://foundry.paradigm.xyz | bash
# Add foundry to PATH and run foundryup
ENV PATH="/root/.foundry/bin:$PATH"
RUN /root/.foundry/bin/foundryup

# Verify Anvil installation
RUN anvil --version

# Clean up build artifacts
WORKDIR /usr/src/app
RUN rm -rf /tmp/solidity_${SOLIDITY_VERSION} /tmp/solidity_${SOLIDITY_VERSION}.tar.gz 
# copy project
COPY . /usr/src/app/

# Copy anvil state file if it exists
RUN chmod +x ./start.sh
CMD ["./start.sh"]