import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import re
import random
import math
import string
import matplotlib.pyplot as plt
# %%
# Read the text file into a DataFrame
AlPoe = pd.read_csv('PycharmProjects/NLP/datasets/allan_poe.txt', delimiter='/n', header=None,engine='python')
RobFrost = pd.read_csv('PycharmProjects/NLP/datasets/robert_frost.txt', delimiter='/n', header=None,engine='python')

# Display the DataFrame
print(AlPoe.shape)
print(RobFrost.shape)
#%% 80- 20 Split

le = len(AlPoe)

AlanTrain, AlanTest  = AlPoe[:int(le*0.8)] , AlPoe[-int(le*0.2):]

le = len(RobFrost)
RobTrain, RobTest = RobFrost[:int(le*0.8)] , RobFrost[-int(le*0.2):]
#%%
with open('PycharmProjects/NLP/datasets/allan_poe.txt', 'r', encoding='utf-8') as file:
    alan = file.read()

with open('PycharmProjects/NLP/datasets/robert_frost.txt', 'r', encoding='utf-8') as file:
    rob = file.read()

    
#alan = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",alan)
#alan = re.sub("[,|!|.|?|\"}\][\']", "", alan)    
#alan = re.sub("\\n", " ", alan)   
#rob = re.sub("\\n", " ", rob)   
#%%
def tokenize(df):
    uniqueWords = []
    # for each row
    for index, row in df.iterrows():
        row = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",str(row.values))
        row = re.sub("[,|!|.|?|\"}\][\']", "", row)
        words = [w.lower() for w in row.split()]

        for word in words:
            if word not in uniqueWords:
                uniqueWords.append(word)
    
    return uniqueWords

def tokenize_txt(txt):
    uniqueWords = {}
    word2vec = []
    vec2word = []
    idx=0
    # remove newline   
    txt = normalize(txt)
    for index, word in enumerate(txt.split(" ")):
        if word not in uniqueWords:
            uniqueWords[word] = idx
            idx += 1
        word2vec.append(uniqueWords.get(word))
        vec2word.append(word)
        
    return uniqueWords, word2vec, vec2word
    
    
    
def UnknownIndex(df,table):
    uniqueWords = []
    # for each row
    for index, row in df.iterrows():
        row = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",str(row.values))
        row = re.sub("[-,|!|.|?|\"}\][\']", "", row)
        words = [w.lower() for w in row.split()]

        for word in words:
            if word not in uniqueWords and word not in table:
                uniqueWords.append(word)
                
    
    return uniqueWords

def cleanup(item):
    item = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",item)
    item = re.sub("[-()-,|!|.|?|\"{}\][\']", "", item)
    return item


def normalize(txt):
    clean_txt = ""
    txt = re.sub("\\n", " ",txt)
    txt = re.sub("\\'", " ",txt)
    txt = txt.replace('\u2009', '')
    for index, word in enumerate(txt.split(" ")):
        word = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",word)
        word = word.translate(str.maketrans('', '', string.punctuation))
        word = re.sub(" {1,}", "", word)
        word = word.lower()
        clean_txt+= word + " "
    return clean_txt

# tokenize sequence
AlanuniqueWords = tokenize(AlanTrain)
RobuniqueWords = tokenize(RobTrain)

AlanUnknownWords = UnknownIndex(AlanTest, AlanTrain)
RobUnknownWords = UnknownIndex(RobTest, RobTrain)


#%%
# correct 1362
un, w2v, vw2 = tokenize_txt(alan)

# should print true 
print(un.get('most')) 
print(w2v[1249])
print(vw2[1249])

#%%

def count_sequence(array, target):
    count = 0
    targ_len = len(target)
    
    for i in range(len(array) - targ_len+1):
        if array[i:i+targ_len] == target:
            count+=1
    
    return count
    



def markov_model(sequence, order=2,eps=1):
    #tokenize sequence
    unique,word2vec,vec2word = tokenize_txt(sequence)
    
    lenght = len(vec2word)
    A = np.empty((lenght, lenght))
    
    markov_model = {}  
    
    #loop from start to end-order
    for i, word in enumerate(vec2word[:-order]):
        con = []
        
        # add next words to context
        context = vec2word[i:i+order+1]

        next_char = vec2word[i+order]
    
        
    
    return markov_model
        



def predict_seq(markov_model, initial_context, length):
    current_context = initial_context
    generated_text = current_context
    for _ in range(length):
        if current_context not in markov_model:
            break
        next_chars = list(markov_model[current_context].keys())
        next_char = random.choice(next_chars)
        generated_text += next_char
        current_context = generated_text[-len(initial_context):]
    return generated_text



#%%

def normalize(txt):
    clean_txt = ""
    txt = re.sub("\\n", " ",txt)
    txt = re.sub("\\'", " ",txt)
    #txt = re.sub("\\u", "",txt)
    for index, word in enumerate(txt.split(" ")):
        word = re.sub(" , [0-2][0-9]:[0-5][0-9]", "",word)
        word = word.translate(str.maketrans('', '', string.punctuation))
        word = re.sub(" {1,}", "", word)
        word = word.lower()
        clean_txt+= word + " "
    return clean_txt

#%% 
normalize(alan)


