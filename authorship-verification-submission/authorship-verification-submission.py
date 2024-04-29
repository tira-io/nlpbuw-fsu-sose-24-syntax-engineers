import re
import nltk
import pandas as pd
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# download nltk tools for text-preprocessing - downloads once
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# helper function to preprocess a string of text
def preprocessText(text):
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

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    # Preprocess the text in training and validation datasets
    text_train['text'] = text_train['text'].apply(preprocessText)
    text_validation['text'] = text_validation['text'].apply(preprocessText)

    # Extract word frequencies as features by vectorization
    vectorizer = TfidfVectorizer()    
    x_train = vectorizer.fit_transform(text_train['text'])

    # Load targets (labels)
    y_train = targets_train['generated']

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Make predictions
    x_validation = vectorizer.transform(text_validation['text'])
    y_pred_validation = model.predict(x_validation)

    # Create DataFrame with 'id' column and add predicted labels 
    predictions = pd.DataFrame(text_validation['id'], columns=['id'])
    predictions['predicted'] = y_pred_validation

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
