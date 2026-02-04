FROM nvcr.io/nvidia/pytorch:24.01-py3
WORKDIR /workspace/EchoPrime
COPY . .
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update
RUN python -m pip install --upgrade pip
RUN python -m pip uninstall opencv-python
RUN python -m pip install --no-cache-dir -r requirements.txt