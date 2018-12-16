# Pull base image
FROM python:3.6-alpine

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set arguments 
ARG PROJECT=server

# Set work directory
RUN mkdir /${PROJECT}
WORKDIR -p /${PROJECT}
COPY . /${PROJECT}

# Install dependencies
RUN pip install pipenv
RUN pipenv install --system --skip-lock

