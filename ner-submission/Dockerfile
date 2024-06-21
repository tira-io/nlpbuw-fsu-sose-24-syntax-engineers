FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD train.py /code/train.py
ADD run.py /code/run.py

RUN pip3 install nltk spacy

ENTRYPOINT [ "python3", "/code/run.py" ]
