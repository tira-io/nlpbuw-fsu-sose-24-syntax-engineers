import spacy
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nlp = spacy.load("en_core_web_sm")
pd.set_option("display.max_rows", 200)
# Getting the pipeline component
ner=nlp.get_pipe("ner")

data = "The 14-kilometer-long tunnel is part of a rail line linking Yichang City in Hubei Province with Wanzhou in southwest China ."
doc = nlp(data)
print("Entities:", doc)
for ent in doc.ents:
        print(ent.text, " - ", ent.start_char," - ", ent.end_char, " - ", ent.label_)

entities = {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}

print(entities)

tuple_data = (data, entities)
print(tuple_data)