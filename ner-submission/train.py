from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

import spacy
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from pathlib import Path


def bio_tagging(nlp, sentence):
    # Tokenize the sentence
    word_tokens = word_tokenize(sentence)
    # print("Word tokens:", word_tokens)

    # Initialize an empty list to store the entities
    bio_tags = ["O"] * len(word_tokens)

    doc = nlp(sentence)
    # print("Entities:", doc)

    # Initialize list to store entities in the required format
    entities = []

    # Track used tokens
    used_tokens = set()

    # for ent in doc.ents:
    #     print(ent.text, " - ", ent.start_char," - ", ent.end_char, " - ", ent.label_)

    tuple_data = ()
    # Check each entity
    for ent in doc.ents: #for entity, tag in entities.items():
        # Tokenize the entity
        entity_tokens = word_tokenize(ent.text)
        entity_len = len(entity_tokens)

        # print("Entity tokens:", entity_tokens)
        
        # Find the entity in the sentence
        for i in range(len(word_tokens) - entity_len + 1):
            if word_tokens[i:i + entity_len] == entity_tokens:
                if i in used_tokens:
                    continue
                start_char = ent.start_char
                for token_index, token in enumerate(entity_tokens):
                    if token_index == 0:
                        tag = f"B-{ent.label_}"
                    else:
                        tag = f"I-{ent.label_}"

                    # Use spaCy's character offsets to find the token position
                    start = start_char
                    end = start + len(token)
                    entities.append((start, end, tag))
                    start_char = end + 1  # Update the start_char for the next token


                    # Mark the token indices as used
                    used_tokens.add(i + token_index)

                break  # Move to the next entity
                
    #print("bio entities:",entities)
        #tuple_data = (doc, entities)

    # spaCy accepts training data as list of tuples.
    # Combine the tokens and bio tags into tuple format
    tuple_data = (sentence, {"entities": entities})
    print("Tuple data:", tuple_data)    

    # Return BIO tags
    #return list(bio_tags)
    return tuple_data

def train_dataset(nlp, sentences):
    train_data = []
    for sentence in sentences:
        tuple_data = bio_tagging(nlp, sentence)
        train_data.append(tuple_data)
    return train_data

if __name__ == "__main__":

    tira = Client()

    # loading val data 
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    nlp = spacy.load("en_core_web_sm")
    pd.set_option("display.max_rows", 200)
    # Getting the pipeline component
    ner=nlp.get_pipe("ner")

    #bio_tagging(text_validation['sentence'][0])
    #print(text_validation.head())
    #print(targets_validation.head())


    TRAIN_DATA = train_dataset(nlp, text_validation['sentence'])
    #print("Train data:", TRAIN_DATA)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            print("label: ", ent[2])
            ner.add_label(ent[2])

    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # TRAINING THE MODEL
    with nlp.disable_pipes(*unaffected_pipes):
        # Training for 30 iterations
        for iteration in range(5):
            print("Iteration #", iteration)
            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            # batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            #for batch in batches:
            for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
                #texts, annotations = zip(*batch)
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], losses=losses, drop=0.3)
                print("Losses", losses)

    model_path = str(Path(__file__).parent) + "/ner_model
    print(model_path)
    # Save the model
    nlp.to_disk(model_path)

    # Load the saved model and test it
    nlp_custom = spacy.load(model_path)

    # Testing the model
    test_data = nlp_custom("The 14-kilometer-long tunnel is part of a rail line linking Yichang City in Hubei Province with Wanzhou in southwest China .")

    for ent in test_data.ents:
        print(ent.text, ent.label_) 

    print("Entities", [(ent.text, ent.label_) for ent in test_data.ents])
