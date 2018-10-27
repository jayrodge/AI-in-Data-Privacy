import spacy
import os
import re
import pandas as pd
from tqdm import tqdm
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en")
path = 'D:/Preperation for Hiring/Data Security Hackathon at IIT/Hackathon_package/documents'

dataset = pd.DataFrame(columns=["id", "text"])
y = pd.read_csv("training_labels.csv", usecols=["id", "score"])
filenames = os.listdir(path)

for index in tqdm(range(len(filenames))):
    filename = filenames[index]
    document = open(path+"/"+filename, encoding='utf-8', errors='ignore').read()
    document = nlp(document)
    filtered_sentence = [word for word in document if word.is_stop == False and word.is_punct == False]
    filtered_sentence = ' '.join(map(str, filtered_sentence))
    filtered_sentence = filtered_sentence.replace(',', '')
    dataset.loc[index] = [int(re.findall('\d+', filename)[0]), str(filtered_sentence).lower()]
    
dataset["id"] = dataset["id"].astype("uint32")
y["id"] = y["id"].astype("uint32")

final_dataset = pd.merge(dataset, y, on='id')
final_dataset["score"] = final_dataset["score"].astype("uint8")

final_dataset.to_csv("final_dataset.csv", encoding='utf-8', index = False)