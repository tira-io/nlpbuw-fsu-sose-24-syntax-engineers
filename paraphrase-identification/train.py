from tira.rest_api_client import Client
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump
from sklearn.linear_model import LogisticRegression
from pathlib import Path

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the sentences separately
    tfidf_matrix_1 = vectorizer.fit_transform(text['sentence1'])
    tfidf_matrix_2 = vectorizer.transform(text['sentence2'])

    # Calculate cosine similarity for each pair of sentences
    text['cossim'] = [cosine_similarity(tfidf_matrix_1[i], tfidf_matrix_2[i])[0][0] for i in range(len(text))]

    x = pd.DataFrame(text['cossim'])
    y = labels['label']

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(x, y)

    # Save the model and vectorizer
    dump(model, Path(__file__).parent / "model.joblib")
    dump(vectorizer, Path(__file__).parent / "vectorizer.joblib")

