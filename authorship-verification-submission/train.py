import re
import nltk
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from tira.rest_api_client import Client
from sklearn.linear_model import LogisticRegression

# download nltk tools for text-preprocessing - downloads once
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def main():
    tira = Client()

    # Loading train data
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")

    # Preprocessing text
    text['text'] = text['text'].apply(preprocess_text)

    # Extract word frequencies as features by vectorization
    vectorizer = CountVectorizer()    
    x = vectorizer.fit_transform(text['text'])
    # Load targets (labels)
    y = labels['generated']

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(x, y)

    # Save the model and vectorizer
    dump(model, Path(__file__).parent / "model.joblib")
    dump(vectorizer, Path(__file__).parent / "vectorizer.joblib")


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