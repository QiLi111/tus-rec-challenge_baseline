FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
MAINTAINER "Qi Li"
WORKDIR /home/tus-rec_challenge/
ENV PATH="/opt/miniconda/bin:${PATH}" 
RUN apt-get -y update \
	&& apt-get -y install wget \
	&& rm -rf /var/lib/apt/lists/* \
	&& wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
	&& apt-get purge -y --auto-remove wget \
	&& apt-get clean \
	&& rm -f  Miniconda3-latest-Linux-x86_64.sh
COPY . .
RUN conda create -n freehand-US python=3.9.13
SHELL ["conda","run","--no-capture-output", "-n","freehand-US","/bin/bash","-c"]
RUN yes | pip install -r requirements.txt \
	&& rm -f requirements.txt
RUN conda install -y pytorch3d --no-deps -c conda-forge
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "freehand-US", "python","test.py"]
