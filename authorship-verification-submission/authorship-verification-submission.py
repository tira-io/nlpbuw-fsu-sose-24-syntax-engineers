import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocessText(text):
    # Define regex pattern to match punctuation and special characters
    pattern = r'[^a-zA-Z\s]'
    # Replace punctuation and special characters with an empty string
    text = re.sub(pattern, '', text)

    # Converting the text to all lowercase to remove case mismatches
    text = text.lower()

    # Tokenizing the text by breaking it up into smaller components (tokens)
    text = word_tokenize(text)
    
    # Stemming the text to remove word affixes (prefixes and suffixes)
    # text = [stemmer.stem(token) for token in text]
    # Stemming is commented out since some words do not make sense with their affixes removed and are ignored by the lemmatizer

    # Lemmatization to bring words down to their root forms
    text = [lemmatizer.lemmatize(token) for token in text]

    # Stopword removal to remove words that don’t provide any additional information
    text = [word for word in text if word not in stop_words] 

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

    print(text_train.head())

    text_train['text'] = text_train['text'].apply(preprocessText)
    text_validation['text'] = text_validation['text'].apply(preprocessText)

    print(text_train.head())

    # Convert list of tokens back to strings
    text_train['text'] = text_train['text'].apply(lambda tokens: ' '.join(tokens))
    text_validation['text'] = text_validation['text'].apply(lambda tokens: ' '.join(tokens))

    # Extract word frequencies as features using CountVectorizer
    vectorizer_word_freq = CountVectorizer()
    x_train = vectorizer_word_freq.fit_transform(text_train['text'])
    x_validation = vectorizer_word_freq.transform(text_validation['text'])

    # Load targets (labels)
    y_train = targets_train['generated']
    y_validation = targets_validation['generated']

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Make predictions
    y_pred_validation = model.predict(x_validation)

    # Calculate F1 score
    f1_validation = f1_score(y_validation, y_pred_validation)
    print("Validation F1 Score:", f1_validation)