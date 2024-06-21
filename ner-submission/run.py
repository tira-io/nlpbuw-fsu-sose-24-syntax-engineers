from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import spacy
from nltk.tokenize import word_tokenize
import pandas as pd


def preprocess_data(sentence, nlp_custom):
    # Process the sentence with the loaded model
    doc = nlp_custom(sentence["sentence"])

    # Extract BIO tags for each token
    tags = []
    for token in doc:
        if token.ent_type_:
            tags.append(token.ent_type_)
        else:
            tags.append("O")
    #print("Tags:", tags)
    return {"id": sentence["id"], "tags": tags}
    


if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    # Convert the loaded validation data to DataFrame
    df_validation = pd.DataFrame(text_validation)

    # Load the saved model
    model_path = get_output_directory(str(Path(__file__).parent.parent) + "/ner-submission/ner_model")
    nlp_custom = spacy.load(model_path)

    # Prepare for predictions
    predictions = []

    # Process each sentence
    for index, row in text_validation.iterrows():
        print("Preprocess has started for the sentence:", row["sentence"])
        # Preprocess and get tags for the sentence
        tags_output = preprocess_data(row, nlp_custom)

        # Append the processed entry
        predictions.append(tags_output)
        #print("Tags Output:", tags_output)

    print("Predictions:", predictions)

    predictions = pd.DataFrame(predictions)
    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
