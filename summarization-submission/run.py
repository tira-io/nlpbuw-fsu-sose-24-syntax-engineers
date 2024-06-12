from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance


def main():
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training")

    # Generate summaries for the dataset
    df["summary"] = df["story"].apply(generate_summary)
    
    df = df.drop(columns=["story"])

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text) # Tokenize text into sentences
    preprocessed_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)  # Tokenize sentence into words
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]  # Lowercase and remove stopwords
        preprocessed_sentences.append(' '.join(words))  # Recreate sentence from processed words
    
    return preprocessed_sentences


def generate_summary(text):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = sent_tokenize(text)

    sentence_similarity_matrix = similarity_matrix(sentences, stop_words) # Build a similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix) # Create a graph from the similarity matrix
    scores = nx.pagerank(sentence_similarity_graph) # Apply the PageRank algorithm
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True) # Rank sentences
    # Pick top 2 sentences for summary
    ranked_sentence_length = len(ranked_sentence)
    for i in range(4):
        if i < ranked_sentence_length:
            summarize_text.append(ranked_sentence[i][1])

    return " ".join(summarize_text)


def similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []

    sent1 = [w.lower() for w in word_tokenize(sent1)]
    sent2 = [w.lower() for w in word_tokenize(sent2)]
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Create word frequency vectors
    for w in sent1:
        if w in stop_words:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in stop_words:
            continue
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)


if __name__ == "__main__":
    main()