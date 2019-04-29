# Dissertation
Repo for ComSci & AI final year project

# Instructions
This project is split into three sections: Training, Backend and Frontend

## Training
In order to run/train any model Python must be installed on your machine. A 'Pipfile' is included to install all the necessary dependencies. To run this Pipfile you will need to install pipenv through the command 'pip install pipenv'. With pipenv installed, to create a new python environment to run any of the models run the command 'pipenv install --skip-lock' and wait. 

NOTE - To run any of the models your computer must have a GPU installed with the necessary CUDA/CUDNN libraries. More information regarding the installation of those libraries can be found here:

## Web App (Frontend and Backend)
Both the frontend and backend of this project are contained within docker containers. To run the web application locally, you will need to install docker and docker-compose if you don't have them already. instructions for installing those can be found here:

After installing docker/docker-compose to run the application locally all you need to do is run the command 'docker-compose up' from the root directory, which will build and start both the frontend and backend containers.
