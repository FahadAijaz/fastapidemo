import re
from nltk.corpus import stopwords

cached_stopwords = set(stopwords.words('english'))
cities =["purchase", "domestic", "gb" ]
cities=list(map(lambda c : c.lower(),cities))
def remove_puntuation(sentence: str):
    punct = re.compile(r'(\w+)+')
    tokenized = [m.group() for m in punct.finditer(sentence)]    
    return ' '.join(tokenized)

def remove_city_names(sentence):
    sentence_split=sentence.split(" ")
    sentence_tokens=[]
    for i in sentence_split:
        if i in cities:
            i=''
        sentence_tokens.append(i)
    sentence = ' '.join(sentence_tokens)
    return sentence


def append_2_single_letters(sentence):
    word_reg = re.compile(r'\w')
    replace_letter = [m.group() for m in word_reg.finditer(sentence)]
    sentence_split=sentence.split(" ")
    sentence_tokens=[]
    for i in sentence_split:
        if i in replace_letter:
            i=i+'xx'
        sentence_tokens.append(i)
    sentence = ' '.join(sentence_tokens)
    return sentence
#%%
def remove_stopword(sentence: str):    
    sentence = ' '.join([word if word not in cached_stopwords else '#' for word in sentence.split()])
    return sentence

def preprocessing_pipeline(df_column):
    df_column = df_column.str.lower()
    df_column = df_column.apply(remove_puntuation)
    df_column =df_column.apply(append_2_single_letters)
    df_column = df_column.apply(remove_city_names)
    df_column = df_column.apply(remove_stopword)
    return df_column

