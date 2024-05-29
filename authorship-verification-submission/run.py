import re
import pandas as pd
from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def main():
    # Load the validation data
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training")

    # Preprocess the text in training and validation datasets
    text['text'] = text['text'].apply(preprocess_text)
    
    # Load the vectorizer and extract features
    vectorizer = load(Path(__file__).parent / "vectorizer.joblib")
    x = vectorizer.transform(text['text'])
    
    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    y_predictions = model.predict(x)

    # Create DataFrame with 'id' column and add predicted labels 
    predictions = pd.DataFrame(text['id'], columns=['id'])
    predictions['generated'] = y_predictions

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )


# Helper function for text preprocessing
def preprocess_text(text):
    # Regex to replace anything that is not alphabet or whitespace with empty string
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Converting the text to all lowercase to remove case mismatches
    text = text.lower()
    # Tokenizing the text by breaking it up into smaller components (tokens)
    text = word_tokenize(text)    
    # Stemming the text to remove word affixes (prefixes and suffixes)
    text = [stemmer.stem(token) for token in text]
    # Lemmatization to bring words down to their root forms
    text = [lemmatizer.lemmatize(token) for token in text]
    # Stopword removal to remove words that don’t provide any additional information
    text = [word for word in text if word not in stop_words] 
    # Join the tokens into a complete string
    text = ' '.join(text)
    return text


if __name__ == "__main__":    
    main()