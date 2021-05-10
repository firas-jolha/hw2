#Set base image python:3.7.10 which is provided by Google Colab 2021
FROM python:3.7.10

#Set the working directory in the container
WORKDIR /code

#Copy the requirements file to working directory in order to install program dependencies
COPY requirements.txt .

#Install the dependencies
RUN pip install -r requirements.txt

RUN mkdir ./src

#Copy the script folder to the working directory 
COPY src/ ./src

RUN mkdir ./data

#Copy the data folder to the working directory
COPY data/ ./data

RUN mkdir ./models

#Copy the models folder to the working directory for testing purposes
COPY models/ ./models

#RUN echo "Training Stage"

#Run the train.py script in the container
CMD ["python", "./src/train.py"]


#RUN echo "Testing Stage"

#Run the test.py script in the container
CMD ["python", "./src/test.py"]
