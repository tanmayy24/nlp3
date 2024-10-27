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
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

def example_transform(example):
    example["text"] = example["text"].lower()
    return example

# QWERTY neighbors map for typo introduction
qwerty_neighbors = {
    'a': ['q', 'w', 's', 'z'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'c'], 'e': ['w', 's', 'd', 'r'], 'f': ['d', 'r', 't', 'g'],
    'g': ['f', 't', 'y', 'h'], 'h': ['g', 'y', 'u', 'j'], 'i': ['u', 'j', 'k', 'o'],
    'j': ['h', 'u', 'i', 'k'], 'k': ['j', 'i', 'o', 'l'], 'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'], 'o': ['i', 'k', 'l', 'p'],
    'p': ['o', 'l'], 'q': ['w', 'a'], 'r': ['e', 'd', 'f', 't'],
    's': ['a', 'w', 'e', 'd', 'x'], 't': ['r', 'f', 'g', 'y'],
    'u': ['y', 'h', 'j', 'i'], 'v': ['c', 'd', 'f', 'b'],
    'w': ['q', 'a', 's', 'e'], 'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'g', 'h', 'u'], 'z': ['a', 's', 'x']
}

# Function to introduce typos
def introduce_typo(word):
    if len(word) > 1 and random.random() < 0.2:  # 20% chance to introduce a typo
        idx = random.randint(0, len(word) - 1)
        if word[idx] in qwerty_neighbors:
            word = word[:idx] + random.choice(qwerty_neighbors[word[idx]]) + word[idx + 1:]
    return word

# Function to apply typo introduction transformation
def typo_introduction(example):
    words = word_tokenize(example["text"])
    new_words = [introduce_typo(word) for word in words]
    example["text"] = TreebankWordDetokenizer().detokenize(new_words)
    return example

# Function to replace words with synonyms
def synonym_replacement(example):
    words = word_tokenize(example["text"])
    new_words = []
    for word in words:
        if random.random() < 0.3:  # 30% chance to replace a word
            synsets = wordnet.synsets(word)
            if synsets:
                # Choose the most common synonym (lemma with the highest usage count)
                lemmas = synsets[0].lemmas()
                best_synonym = max(lemmas, key=lambda lemma: lemma.count()).name()
                new_words.append(best_synonym if best_synonym != word else word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    example["text"] = TreebankWordDetokenizer().detokenize(new_words)
    return example


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE ###

    # Define probability thresholds
    synonym_prob = 0.3  # 30% chance to replace a word with a synonym
    typo_prob = 0.4    # 40% chance to introduce a typo

    words = word_tokenize(example["text"])
    new_words = []
    
    for word in words:
        # Randomly decide if we want to replace the word with a synonym
        if random.random() < synonym_prob:
            synsets = wordnet.synsets(word)
            if synsets:
                synonyms = synsets[0].lemmas()
                if synonyms:
                    new_word = synonyms[0].name()
                    if new_word != word:
                        new_words.append(new_word)
                        continue  # Skip to the next word after replacement

        # If not replaced by a synonym, consider introducing a typo
        if random.random() < typo_prob and len(word) > 1:
            idx = random.randint(0, len(word) - 1)
            if word[idx] in qwerty_neighbors:
                word = word[:idx] + random.choice(qwerty_neighbors[word[idx]]) + word[idx + 1:]
        
        # Add the (possibly modified) word to the result
        new_words.append(word)

    # Reconstruct the transformed text
    example["text"] = TreebankWordDetokenizer().detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
