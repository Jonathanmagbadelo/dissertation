# Pull base image
FROM python:3.6-alpine

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set arguments 
ARG PROJECT=server

# Set work directory
RUN mkdir -p /${PROJECT}
WORKDIR /${PROJECT}

# Install dependencies
RUN pip install pipenv
COPY ./Pipfile /${PROJECT}
RUN pipenv install --system --skip-lock

COPY . /${PROJECT}
