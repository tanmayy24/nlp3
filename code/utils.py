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


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def get_nearest_keys():
    # A mapping of letters to their nearby keys on a QWERTY keyboard
    return {
        'a': ['q', 'w', 's', 'z'],
        'e': ['w', 'r', 's', 'd'],
        'i': ['u', 'o', 'j', 'k'],
        'o': ['i', 'p', 'k', 'l'],
        'u': ['y', 'i', 'h', 'j'],
    }

def introduce_typo(word):
    # Randomly introduce typos in the word by replacing a vowel with one of its nearby keys
    nearest_keys = get_nearest_keys()
    typo_word = list(word)
    for idx, char in enumerate(typo_word):
        if char in nearest_keys and random.random() < 0.2:  # 20% chance to replace the letter
            typo_word[idx] = random.choice(nearest_keys[char])
    return ''.join(typo_word)

def replace_with_synonym(word):
    # Replace word with a synonym if possible
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    # Filter to avoid returning the original word
    if synonyms:
        synonyms.discard(word)
        if synonyms:
            return random.choice(list(synonyms))
    return word

def swap_adjacent_characters(word):
    # Randomly swap adjacent characters in a word
    if len(word) < 2:
        return word
    word_list = list(word)
    idx = random.randint(0, len(word) - 2)  # Choose a position to swap
    word_list[idx], word_list[idx + 1] = word_list[idx + 1], word_list[idx]
    return ''.join(word_list)

def custom_transform(example):
    # Tokenize the input text
    words = word_tokenize(example["text"])
    transformed_words = []

    for word in words:
        # Randomly decide to introduce a typo, replace with a synonym, or swap characters
        if random.random() < 0.2:  # 10% chance to introduce a typo
            word = introduce_typo(word)
        if random.random() < 0.2:  # 10% chance to replace with a synonym
            word = replace_with_synonym(word)
        if random.random() < 0.2:  # 10% chance to swap adjacent characters
            word = swap_adjacent_characters(word)
        transformed_words.append(word)
    
    # Detokenize back to string
    example["text"] = TreebankWordDetokenizer().detokenize(transformed_words)
    return example
