FROM ubuntu:18.04

ENV CERES_VERSION="1.14.0"
ENV SOPHUS_COMMIT="a0fe89a323e20c42d3cecb590937eb7a06b8343a"
ENV CATKIN_WS=/root/catkin_ws
ENV DEBIAN_FRONTEND=noninteractive

# Install ROS Melodic from packages.ros.org
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-melodic-ros-base \
    ros-melodic-cv-bridge \
    ros-melodic-image-transport \
    ros-melodic-message-filters \
    ros-melodic-tf && \
    rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=melodic
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    libatlas-base-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    python-catkin-tools \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-tf && \
    rm -rf /var/lib/apt/lists/*

# Build and install Ceres (separate layer for caching)
COPY ceres-1.14.0.tar.gz /tmp/ceres.tar.gz
RUN if [ "x$(nproc)" = "x1" ] ; then export USE_PROC=1 ; \
    else export USE_PROC=$(($(nproc)/2)) ; fi && \
    cd /tmp && \
    tar xzf ceres.tar.gz && \
    mv ceres-solver-${CERES_VERSION} ceres-solver && \
    cd ceres-solver && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(USE_PROC) install && \
    rm -rf /tmp/ceres-solver /tmp/ceres.tar.gz

# Build and install Sophus (separate layer for caching)
COPY sophus.tar.gz /tmp/sophus.tar.gz
RUN if [ "x$(nproc)" = "x1" ] ; then export USE_PROC=1 ; \
    else export USE_PROC=$(($(nproc)/2)) ; fi && \
    cd /tmp && \
    tar xzf sophus.tar.gz && \
    mv Sophus-${SOPHUS_COMMIT} Sophus && \
    cd Sophus && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(USE_PROC) install && \
    rm -rf /tmp/Sophus /tmp/sophus.tar.gz

# Create workspace directory
RUN mkdir -p $CATKIN_WS/src/VIW-Fusion/

WORKDIR $CATKIN_WS
ENV TERM xterm
ENV PYTHONIOENCODING UTF-8

# Entry point: source ROS + catkin workspace
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/ros/melodic/setup.bash' >> /entrypoint.sh && \
    echo 'if [ -f /root/catkin_ws/devel/setup.bash ]; then source /root/catkin_ws/devel/setup.bash; fi' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
