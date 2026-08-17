FROM ros:humble

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    ros-humble-dynamixel-sdk \
    python3-colcon-common-extensions \
    python3-pip \
    nano \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pigpio pyserial

WORKDIR /root/ws

CMD ["bash"]

