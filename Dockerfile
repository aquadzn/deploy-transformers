FROM pytorch/pytorch:latest

RUN pip install transformers deploy-transformers

WORKDIR /workspace

COPY templates /checkpoint
COPY static /static

RUN apt-get clean && apt-get -y autoremove && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*