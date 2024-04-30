# docker build -t fschlatt/authorship-verification-submission:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD model.joblib /code/model.joblib
ADD vectorizer.joblib /code/vectorizer.joblib

RUN python3 -m nltk.downloader punkt wordnet stopwords

ENTRYPOINT [ "python3", "/code/run.py" ]
