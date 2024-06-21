FROM fschlatt/natural-language-processing-exercises:0.0.1

#ADD train.py /code/train.py
#ADD run.py /code/run.py

# Set working directory inside the container
WORKDIR /code

# Copy the entire ner-submission directory to /code in the container
COPY . /code

RUN pip3 install nltk spacy

# Add the NER model directory to the Docker image
ADD ner_model /code/ner_model

ENTRYPOINT [ "python3", "/code/run.py" ]
