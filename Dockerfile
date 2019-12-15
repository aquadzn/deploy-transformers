FROM python:3.7-slim-stretch

RUN apt-get update

WORKDIR /
COPY templates /checkpoint
COPY static /static

RUN pip --no-cache-dir install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install starlette

RUN apt-get clean && apt-get -y autoremove && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*