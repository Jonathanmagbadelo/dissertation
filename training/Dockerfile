FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ADD install.sh /tmp/install.sh
RUN sh /tmp/install.sh

# Arguments 
ARG PROJECT=training

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off

ENV CUDA_HOME /usr/lib/x86_64-linux-gnu

RUN apt-get update
RUN apt-get install -y software-properties-common

# Install Java.
RUN \
  echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update && \
  apt-get install -y oracle-java8-installer && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /var/cache/oracle-jdk8-installer

# Define commonly used JAVA_HOME variable
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

RUN curl https://raw.githubusercontent.com/kennethreitz/pipenv/master/get-pipenv.py | python3

RUN mkdir -p /opt/${PROJECT}
COPY ./Pipfile /opt/${PROJECT}
WORKDIR /opt/${PROJECT}
RUN pipenv install --system --skip-lock

RUN pip3 install --upgrade http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install --upgrade pyro-ppl torchvision

# ADD xgboost.sh /tmp/xgboost.sh
# RUN sh /tmp/xgboost.sh

RUN pip3 install --upgrade git+git://github.com/AmazaspShumik/sklearn-bayes.git

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

COPY run_jupyter.sh /opt/${PROJECT}
COPY . /opt/${PROJECT}
