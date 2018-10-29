import spacy
import os
import re
import pandas as pd
from tqdm import tqdm
from functions import *
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en")
path = 'D:/Preperation for Hiring/Data Security Hackathon at IIT/Mission-Mars/documents'

dataset = pd.DataFrame(columns=["id", "text"])
y = pd.read_csv("training_labels.csv", usecols=["id", "score"])
filenames = os.listdir(path)

replacements = {' .':'.', " 's":"", '(s)':'', '(s':'', '   ':' '}

for index in tqdm(range(len(filenames))):
    filename = filenames[index]
    document = open(path+"/"+filename, encoding='utf-8', errors='ignore').read()
    document = nlp(document)
    filtered_sentence = [word for word in document if word.text in {'.', '?'} or word.is_stop is False and word.is_punct == False]
    filtered_sentence = ' '.join(map(str, filtered_sentence))
    filtered_sentence = multi_replace(filtered_sentence, replacements)
    filtered_sentence = str(filtered_sentence).lower()
    dataset.loc[index] = [int(re.findall('\d+', filename)[0]), filtered_sentence]

dataset["id"] = dataset["id"].astype("uint32")
y["id"] = y["id"].astype("uint32")
final_dataset = pd.merge(dataset, y, on='id')

final_dataset = final_dataset.drop_duplicates(subset=["text"])

final_dataset['word_count'] = final_dataset['text'].apply(lambda x: word_count(x))
final_dataset['sentence_count'] = final_dataset['text'].apply(lambda x: sentence_count(x))
final_dataset['avg_sentence_length'] = final_dataset['word_count'].astype("float")/final_dataset['sentence_count'].astype("float")
final_dataset['syllables_count'] = final_dataset['text'].apply(lambda x: syllables_count(x))
final_dataset['avg_syllables_per_word'] = final_dataset['text'].apply(lambda x: avg_syllables_per_word(x))
final_dataset['difficult_words'] = final_dataset['text'].apply(lambda x: difficult_words(x))
final_dataset['poly_syllable_count'] = final_dataset['text'].apply(lambda x: poly_syllable_count(x))
final_dataset['flesch_reading_ease'] = final_dataset['text'].apply(lambda x: flesch_reading_ease(x))
final_dataset['gunning_fog'] = final_dataset['text'].apply(lambda x: gunning_fog(x))
final_dataset['smog_index'] = final_dataset['text'].apply(lambda x: smog_index(x))
final_dataset['dale_chall_readability_score'] = final_dataset['text'].apply(lambda x: dale_chall_readability_score(x))

minor_synonyms = set(['minor', 'child', 'kid', 'youngster'])
geolocation_synonyms = set(['geo-location', 'geolocation', 'location', 'position'])
vendor_synonyms = set(['vendor', 'vender', 'seller', 'marketer'])

final_dataset['number_of_email_addresses'] = final_dataset['text'].apply(lambda x: number_of_email_addresses(x))
final_dataset['use_cookies'] = final_dataset['text'].apply(lambda x: use_cookies(x))

# final_dataset['is_minor'] = final_dataset['text'].apply(lambda x: is_minor(x))
# final_dataset['is_how_collect'] = 
# final_dataset['is_geo_location'] = 
# final_dataset['is_vendor'] = 
# final_dataset['is_not_sell'] = 
# final_dataset['is_sell'] = 
# final_dataset['is_share'] = 
# final_dataset['is_not_share'] = 

final_dataset.to_csv("final_dataset.csv", encoding='utf-8', index = False)