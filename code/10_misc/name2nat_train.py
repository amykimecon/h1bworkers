import os
os.chdir('/Users/amykim/Documents/GitHub/h1bworkers/code/misc')

import random
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import OneHotEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import pickle

os.makedirs('data', exist_ok=True)

def convert(name_f, nat_f, fout):
    with open(fout, 'w', encoding='utf8') as fout:
        names = open(name_f, 'r').read().strip().splitlines()
        nats = open(nat_f, 'r').read().strip().splitlines()
        for name, nat in zip(names, nats):
            if "train" in name_f and nat == "Korean":
                if random.random() > 0.5:
                    name = name.replace("-", "")
                if random.random() > 0.5:
                    columns = name.split(" ", 1)
                    if len(columns)==2:
                        last, first = columns
                        name = first + " " + last
            name = name.replace(" ", "▁")
            name = " ".join(char for char in name)
            fout.write(f"{name}\t{nat}\n")

convert('nana/train.src', 'nana/train.tgt', 'data/train.txt')
convert('nana/dev.src', 'nana/dev.tgt', 'data/dev.txt')

# this is the folder in which train, test and dev files reside
data_folder = 'data'

# column format indicating which columns hold the text and label(s)
column_name_map = {0: "text", 1: "label"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         train_file="train.txt",
                                         dev_file="dev.txt",
                                         skip_header=False,
                                         delimiter='\t',    # tab-separated files
                                         label_type = 'class'
)

# stats = corpus.obtain_statistics()
# print(stats)

# create the label dictionary
label_dict = corpus.make_label_dictionary(label_type='class')
#print(label_dict)

# make a list of word embeddings
#embeddings = [OneHotEmbeddings(vocab_dictionary=label_dict)]
from flair.embeddings import WordEmbeddings
embeddings = [WordEmbeddings('glove')]

# initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings = DocumentRNNEmbeddings(embeddings, bidirectional=True, hidden_size=256)

# create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type='class')

# initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

try:
    trainer.train('resources/',
                  learning_rate=0.1,
                  mini_batch_size=128,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=20)
except IndexError as e:
    print("🔥 Crashed with IndexError:", e)
    for sentence in corpus.train:
        for label in sentence.labels:
            if label.value not in label_dict.get_items():
                print("Offending label:", label.value)
                
pickle.dump(label_dict, open('label_dict.pkl','wb'))

