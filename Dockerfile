#
FROM python:3.7.10

#
WORKDIR /code

COPY requirements.txt .


RUN pip install -r requirements.txt


COPY src/ .


RUN echo "Training Stage"

CMD ['python', './train.py']


RUN echo "Testing Stage"

CMD ['python', './test.py']
