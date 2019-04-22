FROM python:3.6-slim


# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
#ENV PIP_NO_CACHE_DIR off


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
