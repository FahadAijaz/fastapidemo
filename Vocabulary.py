from nltk import ngrams
import re

def record_unigram_vocabulary(dataframe):
    unigram_ls = []
    
    for i, df_row in enumerate(dataframe['Transaction Description']): 
        unigrams = list(map(lambda x: x[0], ngrams(df_row.split(' '), n=1)))
        unigram_ls = unigrams + unigram_ls
    regex_pattern = re.compile(r'\d+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d*|[+-]?(\d*[.])?\d+', )
    unfiletered_ls = list(set(unigram_ls))
    return [i for i in unfiletered_ls if not regex_pattern.match(i)]

def char_ngram(unigram_vocab, n = 2):
    char_ngram_vocab_list =[]
    for word in unigram_vocab:
        char_ngram_vocab_list.append([word[i:i+n] for i in range(len(word)-n+1)])
    char_ngram_vocab = [item for sublist in char_ngram_vocab_list for item in sublist]
    return char_ngram_vocab

def record_2letter_words(unigram_vocab):
    letter2_ls = []
    for i in unigram_vocab:
        if len(i) == 2:
            letter2_ls.append(i)
    return letter2_ls
