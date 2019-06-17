FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER zheey "zhezheey@yeah.net"

ADD resources/sources.list /etc/apt/
ADD resources/pip.conf /root/.pip/

RUN apt-get update && \
apt-get install -y --no-install-recommends build-essential curl libfreetype6-dev libpng12-dev libzmq3-dev pkg-config python python-dev python3.5-dev software-properties-common && \
apt-get clean

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
python3 get-pip.py && \
rm get-pip.py

RUN pip3 --no-cache-dir install h5py tensorflow-gpu==1.4.0 Keras==2.0.8

ADD resources/predict_part1.py /data/
ADD resources/predict_part2.py /data/
ADD resources/predict_part3.py /data/
ADD resources/merge.py /data/
ADD resources/ensemble.py /data/
ADD resources/submit.sh /data/
ADD resources/submit_1_0_200.hdf5 /data/
ADD resources/submit_1_20_200.hdf5 /data/
ADD resources/submit_1_40_200.hdf5 /data/
ADD resources/submit_1_0_60.hdf5 /data/
ADD resources/submit_2_0_200.hdf5 /data/
ADD resources/submit_2_20_200.hdf5 /data/
ADD resources/submit_2_40_200.hdf5 /data/
ADD resources/submit_2_0_60.hdf5 /data/
ADD resources/submit_3_0_200.hdf5 /data/
ADD resources/submit_3_20_200.hdf5 /data/
ADD resources/submit_3_40_200.hdf5 /data/
ADD resources/submit_3_0_60.hdf5 /data/
ADD resources/submit_4_0_200.hdf5 /data/
ADD resources/submit_4_20_200.hdf5 /data/
ADD resources/submit_4_40_200.hdf5 /data/
ADD resources/submit_4_0_60.hdf5 /data/
ADD resources/submit_5_0_200.hdf5 /data/
ADD resources/submit_5_20_200.hdf5 /data/
ADD resources/submit_5_40_200.hdf5 /data/
ADD resources/submit_5_0_60.hdf5 /data/
ADD resources/submit_6_0_200.hdf5 /data/
ADD resources/submit_6_20_200.hdf5 /data/
ADD resources/submit_6_40_200.hdf5 /data/
ADD resources/submit_6_0_60.hdf5 /data/
ADD resources/submit_7_0_200.hdf5 /data/
ADD resources/submit_7_20_200.hdf5 /data/
ADD resources/submit_7_40_200.hdf5 /data/
ADD resources/submit_7_0_60.hdf5 /data/
ADD resources/submit_8_0_200.hdf5 /data/
ADD resources/submit_8_20_200.hdf5 /data/
ADD resources/submit_8_40_200.hdf5 /data/
ADD resources/submit_8_0_60.hdf5 /data/
ADD resources/submit_9_0_200.hdf5 /data/
ADD resources/submit_9_20_200.hdf5 /data/
ADD resources/submit_9_40_200.hdf5 /data/
ADD resources/submit_9_0_60.hdf5 /data/
ADD resources/submit_10_0_200.hdf5 /data/
ADD resources/submit_10_20_200.hdf5 /data/
ADD resources/submit_10_40_200.hdf5 /data/
ADD resources/submit_10_0_60.hdf5 /data/

CMD sh /data/submit.sh >/data/logs/log 2>&1