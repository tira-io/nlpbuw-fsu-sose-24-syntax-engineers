from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    )

    # Load the TF-IDF vectorizer
    vectorizer = load(Path(__file__).parent / "vectorizer.joblib")

    # Fit and transform the sentences separately
    tfidf_matrix_1 = vectorizer.transform(df['sentence1'])
    tfidf_matrix_2 = vectorizer.transform(df['sentence2'])

    # Calculate cosine similarity for each pair of sentences
    df['cossim'] = [cosine_similarity(tfidf_matrix_1[i], tfidf_matrix_2[i])[0][0] for i in range(len(df))]
    
    x = pd.DataFrame(df['cossim'])

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    y_predictions = model.predict(x)
    
    # Create DataFrame with 'id' column and add predicted labels 
    predictions = pd.DataFrame(df['id'], columns=['id'])
    predictions['label'] = y_predictions

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )