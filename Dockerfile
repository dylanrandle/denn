FROM jupyter/datascience-notebook

# Add our source code and setup file
ARG PROJECT_ROOT="."
ARG PROJECT_MOUNT_DIR="/denn"
ADD $PROJECT_ROOT $PROJECT_MOUNT_DIR

# cd into project
WORKDIR $PROJECT_MOUNT_DIR

# Install our code
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install .
