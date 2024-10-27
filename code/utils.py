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

# Synonym replacement using WordNet
def synonym_replace(text, replace_prob=0.3):
    words = word_tokenize(text)
    new_words = []
    
    for word in words:
        if random.random() < replace_prob:
            synonyms = wordnet.synsets(word)
            if synonyms:
                syn_word = synonyms[0].lemmas()[0].name()
                new_words.append(syn_word if syn_word != word else word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    
    return TreebankWordDetokenizer().detokenize(new_words)

# Typo introduction by simulating keyboard mistakes
keyboard_typos = {
    'a': ['q', 'w', 's', 'z'], 'e': ['w', 'r', 'd'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
    's': ['a', 'd', 'w', 'x'], 't': ['r', 'y', 'g'], 'n': ['b', 'm', 'j'], 'm': ['n', 'j', 'k']
}

def introduce_typos(text, typo_prob=0.2):
    chars = list(text)
    for i, char in enumerate(chars):
        if char.lower() in keyboard_typos and random.random() < typo_prob:
            chars[i] = random.choice(keyboard_typos[char.lower()])
    return "".join(chars)

# Transformations
def contractions_transform(text):
    for phrase, contraction in contractions.items():
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', contraction, text, flags=re.IGNORECASE)
    return text

def decontractions_transform(text):
    for contraction, phrase in decontractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', phrase, text, flags=re.IGNORECASE)
    return text

# Custom transform function that applies random transformation
def custom_transform(example, transform_probs={"contraction": 0.25, "decontraction": 0.25, "typo": 0.25, "synonym": 0.25}):
    transform_type = random.choices(
        list(transform_probs.keys()), 
        list(transform_probs.values())
    )[0]
    
    if transform_type == "contraction":
        example["text"] = contractions_transform(example["text"])
    elif transform_type == "decontraction":
        example["text"] = decontractions_transform(example["text"])
    elif transform_type == "typo":
        example["text"] = introduce_typos(example["text"])
    elif transform_type == "synonym":
        example["text"] = synonym_replace(example["text"])
    
    return example

