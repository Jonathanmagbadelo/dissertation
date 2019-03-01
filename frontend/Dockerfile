FROM node:8.7.0-alpine

ARG PROJECT=client

# Create project directory 
RUN mkdir -p /${PROJECT}
WORKDIR /${PROJECT}

COPY ./package.json /${PROJECT}

RUN npm install

COPY . /${PROJECT}
