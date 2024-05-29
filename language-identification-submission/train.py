from pathlib import Path
import re
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
current_dir = Path(__file__).parent
# print(current_dir.parent)
stop_words_dir = current_dir.parent / "language-identification-stopwords"/ "stopwords"
lang_ids = [
        "af",
        "az",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hr",
        "it",
        "ko",
        "nl",
        "no",
        "pl",
        "ru",
        "ur",
        "zh",
    ]

stopwords = {
    lang_id: set(
        (stop_words_dir/ f"stopwords-{lang_id}.txt")
        .read_text()
        .splitlines()
    )
    - set(("(", ")", "*", "|", "+", "?"))  # remove regex special characters
    for lang_id in lang_ids
}

# text preprocess
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
    text = [word for word in text if word not in stopwords] 
    # Join the tokens into a complete string
    text = ' '.join(text)
    return text

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-train-20240429-training")
    targets_train = tira.pd.truths("nlpbuw-fsu-sose-24", "language-identification-train-20240429-training")

    text_train = text_train.set_index("id")
    df = text_train.join(targets_train.set_index("id"))

    print("Language Count: ", targets_train["lang"].value_counts())

    print("Merged Dataframe: ", df.head())

    # Preprocessing text
    df['text'] = df['text'].apply(preprocess_text)
    print("Preprocessed Dataframe: ", df.head())

    # convert sentences into a vector
    cv = CountVectorizer(max_features=9999)
    X = cv.fit_transform(df["text"]).toarray()
    print("X shape: ", X.shape)


    # convert categorical data to numerical data
    le = LabelEncoder()
    y = le.fit_transform(df["lang"])
    print("y shape: ", y.shape)
    print("y: ", y)
    print("y classes: ", le.classes_)
    print(type(y))

    final_data = pd.DataFrame(np.c_[df["text"], y], columns=["text", "lang"])
    print("Final data:\n", final_data.head())


    # Train the model
    model = Pipeline(
        [("vectorizer", cv), ("classifier", MultinomialNB())]
    )

    model.fit(df["text"], df["lang"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")


    #prediction scores:
    test_text = tira.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")
    targets_test = tira.pd.truths("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")

    test_text = test_text.set_index("id")
    test_df = test_text.join(targets_test.set_index("id"))

    print("Test Data: ", test_df.head())
    pred = model.predict(test_df["text"])
    print("Predictions: ", pred)
    print("Labels: ", test_df["lang"])
    print("Accuracy Score: ", accuracy_score(test_df["lang"], pred))
    print("Confusion Matrix: ", confusion_matrix(test_df["lang"], pred))

