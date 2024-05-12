FROM fschlatt/natural-language-processing-exercises:0.0.2

ADD run.py /code/run.py
ADD model.joblib /code/model.joblib

RUN pip3 install spacy \
    && python3 -m spacy download en_core_web_sm

ENTRYPOINT ["python3", "/code/run.py"]