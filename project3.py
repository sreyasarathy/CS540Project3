## Written by: Sreya Sarathy
## Attribution: Hugh Liu's solutions for CS540 2021 Epic
## Comments added by: Naman Gupta
## Collaborated with Harshet Anand from CS540

## We use the following import statements in this project
import string
import re
from collections import Counter
from itertools import product
from itertools import permutations
import random
from numpy import cumsum
import numpy as np
from numpy.ma import log

# adjust on your own
# my values were 0.83, 0.17, 1000
P_my = 0.83
P_fake = 0.17
num_charactors = 1000

# The script is stored in thor.txt because I chose the movie
# Thor - Ragnarok
with open("thor.txt", encoding="utf-8") as f:
    data = f.read()

# This function preprocesses the text data
def process_text(data):
    ''' Preprocess the text data '''
    data = data.lower()
    data = re.sub(r"[^a-z ]+", "", data)
    data = " ".join(data.split())
    data = re.sub(' +', ' ', data)

    return data

processed_data = process_text(data)

with open('newthor.txt', 'w') as f:
    f.write(processed_data)

data = process_text(data)

# all possible characters
allchar = " " + string.ascii_lowercase

# The following lines of code deal with the unigram
unigram = Counter(data)
unigram_prob = {ch: round(unigram[ch] / len(data), 4) for ch in allchar}
uni_list = [unigram_prob[c] for c in allchar]

# We now need to distinguish between fake_unigram_prob below
my_unigram_prob = unigram_prob


# The following lines of code generate n-gram
def ngram(n):
    ''' Generate n-gram '''
    # all possible n-grams
    d = dict.fromkeys(["".join(i) for i in product(allchar, repeat=n)], 0)
    # update counts
    d.update(Counter(data[x : x + n] for x in range(len(data) - 1)))
    return d

# We need to deal with bigram now
bigram = ngram(2)
bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}
bigram_prob_L = {c: (bigram[c] + 1) / (unigram[c[0]] + 27) for c in bigram}

# We need to deal with trigram now as well
trigram = ngram(3)
trigram_prob_L = {c: (trigram[c] + 1) / (bigram[c[:2]] + 27) for c in trigram}


# The following lines are based on https://python-course.eu/numerical-programming/weighted-probabilities.php
# The function below randomly chooses an element from collection according to weights.
def weighted_choice(collection, weights):
    """Randomly choose an element from collection according to weights"""
    weights = np.array(weights)
    weights_sum = weights.sum()
    weights = weights.cumsum() / weights_sum
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return collection[i]


# The function below generates the second char
def gen_bi(c):
    ''' Generate the second char '''
    w = [bigram_prob[c + i] for i in allchar]
    return weighted_choice(allchar, weights=w)[0]


# The function below generates the third char
def gen_tri(ab):
    ''' Generate the third char '''
    w = [trigram_prob_L[ab + i] for i in allchar]
    return weighted_choice(allchar, weights=w)[0]


# To generate the second char
def gen_sen(c, num):
    ''' generate the second char'''
    res = c + gen_bi(c)
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
    return res


# generate sentences
sentences = []
for char in allchar:
    sentence = gen_sen(char, num_charactors)
    sentences.append(sentence)

## The following lines of code deal with the fake script
with open("script.txt", encoding="utf-8") as f:
    data = f.read()

data = process_text(data)

unigram = Counter(data)
unigram_prob = {ch: round(unigram[ch] / len(data), 4) for ch in allchar}
uni_list = [unigram_prob[c] for c in allchar]

fake_unigram_prob = unigram_prob


count = 0
for char in allchar:
    count += 1
    print(
        P_fake
        * fake_unigram_prob[char]
        / (P_fake * fake_unigram_prob[char] + P_my * my_unigram_prob[char])
    )

# print(count)

for sentence in sentences:
    my = log(P_my)
    fake = log(P_fake)
    for char in sentence:
        my += np.log10(my_unigram_prob[char])
        fake += np.log10(fake_unigram_prob[char])
    if my > fake:
        print("0")
    else:
        print("1")


# The following lines are used in question 2 of the project.
def print_unigram_prob(unigram_prob_dict):
    all_chars = sorted(unigram_prob_dict.keys())
    print(', '.join(f"{unigram_prob_dict[ch]:.4f}" for ch in all_chars))

print("Unigram Probabilities for original script:")
print_unigram_prob(my_unigram_prob)

# The following lines of code are used in Question 3
# We need to calculate and print bigram transition probabilities
print("Without Soothing:")
for ch1 in allchar:
    bigram_probs = [round(bigram_prob[ch1 +ch2], 4) for ch2 in allchar]


    print(", ".join(f"{prob:.4f}" for prob in bigram_probs))


# The following lines of code are used in Question 4
print("With Soothing:")

# Loop through each character (char1) in the 'allchar' list.
for char1 in allchar:
    line = []
    # Nested loop: Loop through each character (char2) in the 'allchar' list again.
    for char2 in allchar:
        prob = bigram_prob_L.get(char1 + char2, 0)
        line.append("{:.4f}".format(prob))
    print(", ".join(line))

# The following lines of code are used in Question 5
sentences = []

# Loop through each lowercase character in the ASCII alphabet.
for char in string.ascii_lowercase:
    sentence = gen_sen(char, num_charactors)
    sentences.append(sentence)

# Print the sentences
for sentence in sentences:
    print(sentence)


# The following lines of code are used in Question 6
# Initialize an empty list to store the generated sentences.
sentences = []

# Generate sentences for each lowercase character using a custom function 'gen_sen'.
# 'gen_sen' is a function that takes a character 'char' and the number of characters 'num_charactors' as input.
# It generates a sentence based on the given character and the specified number of characters.
# The sentences are then appended to the 'sentences' list.
for char in string.ascii_lowercase:
     sentence = gen_sen(char, num_charactors)
     sentences.append(sentence)

# Print the generated sentences for each lowercase character along with their respective letter.
for i, sentence in enumerate(sentences):
    print(f"Sentence for letter '{string.ascii_lowercase[i]}': {sentence}\n")

# The following lines of code are used in question 7
# We start off by following steps:
# Define a function to print the unigram probabilities from a given dictionary.
def print_unigram_prob(unigram_prob_dict):
    all_chars = sorted(unigram_prob_dict.keys())
    print(', '.join(f"{unigram_prob_dict[ch]:.4f}" for ch in all_chars))

# Call the 'print_unigram_prob' function with the 'fake_unigram_prob' dictionary as input.
# This will print the unigram probabilities for characters in the 'fake' script.
print("Unigram Probabilities for fake script (Question 2 Answer):")
print_unigram_prob(fake_unigram_prob)

# The following lines of code are used in question 8.
# Initialize an empty list to store the posterior probabilities for each character.
posterior_probs = []

for char in allchar:
    P_my_char = my_unigram_prob[char]
    P_fake_char = fake_unigram_prob[char]

    # Calculate the overall probability of the character across both classes using Bayes' rule.
    # 'P_my' and 'P_fake' are predefined probabilities for the 'my' and 'fake' classes, respectively.
    P_char = P_my * P_my_char + P_fake * P_fake_char

    P_fake_given_char = (P_fake_char * P_fake) / P_char

    # Append the rounded posterior probability to the 'posterior_probs' list.
    posterior_probs.append(round(P_fake_given_char, 4))

# Output the computed posterior probabilities for the characters in the 'allchar' list.
print("Unigram Probabilities for fake script:")
print(', '.join(f"{prob:.4f}" for prob in posterior_probs))

# The following lines of code are used in question 9.
# This code performs a binary classification task using log probabilities.
# It processes a list of 'sentences' and assigns a label of either 0 (indicating "my" sentence)
# or 1 (indicating "fake" sentence) to each sentence.
# The 'predictions' list will store the resulting labels for each sentence.
predictions = []

for sentence in sentences:
    log_prob_my = np.log(P_my)
    log_prob_fake = np.log(P_fake)

    for char in sentence:
        log_prob_my += np.log(my_unigram_prob[char])
        log_prob_fake += np.log(fake_unigram_prob[char])

    predictions.append(0 if log_prob_my > log_prob_fake else 1)

print(predictions)