import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import re
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# Define a dictionary of common contractions and their expanded forms
contractions = {
    "do not": "don't", "I am": "I'm", "you are": "you're", "we are": "we're", "they are": "they're",
    "is not": "isn't", "are not": "aren't", "cannot": "can't", "will not": "won't", "did not": "didn't",
    "would not": "wouldn't", "should not": "shouldn't", "could not": "couldn't", "have not": "haven't",
    "has not": "hasn't", "had not": "hadn't", "it is": "it's", "that is": "that's", "there is": "there's",
    "what is": "what's", "who is": "who's", "let us": "let's", "we will": "we'll", "you will": "you'll",
    "they will": "they'll", "he will": "he'll", "she will": "she'll", "I will": "I'll", "it will": "it'll",
    "I would": "I'd", "you would": "you'd", "he would": "he'd", "she would": "she'd", "we would": "we'd",
    "they would": "they'd", "there are": "there're", "I have": "I've", "you've": "you have", "we've": "we have",
    "they've": "they have", "that'd": "that would", "who'd": "who would", "where's": "where is",
    "how's": "how is", "when's": "when is", "why's": "why is"
}

# Reverse mapping for decontractions
decontractions = {v: k for k, v in contractions.items()}

def contractions_transform(text):
    # Replace phrases with their contracted forms
    for phrase, contraction in contractions.items():
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', contraction, text, flags=re.IGNORECASE)
    return text

def decontractions_transform(text):
    # Replace contractions with their expanded forms
    for contraction, phrase in decontractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', phrase, text, flags=re.IGNORECASE)
    return text

def custom_transform(example, contraction_prob=0.5):
    ################################
    ##### YOUR CODE BEGINS HERE ###

    # Randomly decide whether to apply contraction or decontraction based on the given probability
    if random.random() < contraction_prob:
        example["text"] = contractions_transform(example["text"])
    else:
        example["text"] = decontractions_transform(example["text"])

    ##### YOUR CODE ENDS HERE ######

    return example


# Example usage
example_text = {"text": "I am not sure if you are going to like it."}
random.seed(42)  # For reproducibility
print(custom_transform(example_text))

example_text_2 = {"text": "I'm not sure if you're going to like it."}
print(custom_transform(example_text_2))