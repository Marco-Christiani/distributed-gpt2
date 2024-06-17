# Use an ARM compatible base image
FROM arm32v7/ubuntu:latest

# Install basic tools and OpenBLAS dependencies
RUN apt-get update && apt-get install -y build-essential gfortran git libopenblas-base libopenblas-dev

# Clone the OpenBLAS repository
RUN git clone https://github.com/xianyi/OpenBLAS.git /opt/OpenBLAS

# Build OpenBLAS
RUN cd /opt/OpenBLAS && \
    make BINARY=32 FC=gfortran USE_THREAD=1 NO_AFFINITY=1 NUM_THREADS=32 && \
    make PREFIX=/opt/OpenBLAS/install install

# Set environment variables
ENV LD_LIBRARY_PATH=/opt/OpenBLAS/install/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/OpenBLAS/install/lib:$LIBRARY_PATH
ENV C_INCLUDE_PATH=/opt/OpenBLAS/install/include:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=/opt/OpenBLAS/install/include:$CPLUS_INCLUDE_PATH
