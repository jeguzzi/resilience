FROM python:3

MAINTAINER Jerome Guzzi "jerome@idsia.ch"

WORKDIR /resilient_traversability

COPY code /resilient_traversability/code

RUN pip install -r code/requirements.txt
