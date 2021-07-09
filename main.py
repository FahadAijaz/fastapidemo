from SimariltyResult import SimilarityScore, SimilarityScoreCollection
from FindClosest import find_max_simialrity
from Vocabulary import record_unigram_vocabulary,char_ngram, record_2letter_words
from fastapi import FastAPI
from typing import List
from TestRow import TestRow
from TrainRow import TrainRow
from Preprocessing import preprocessing_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import re
app = FastAPI()
# initilaized

@app.post("/", response_model=SimilarityScoreCollection)
async def infer_submerchant(train_description_rows: List[str], train_submerchant_rows: List[str], test_rows: List[str]):
    train_df = pd.DataFrame({
        'Transaction Description': train_description_rows,
        'SubMerchant': train_submerchant_rows
        })
    test_df = pd.DataFrame({
        'Transaction Description': test_rows
        })
    train_df ["Transaction Description"]= preprocessing_pipeline(train_df["Transaction Description"])
    test_df ["Transaction Description"]= preprocessing_pipeline(test_df["Transaction Description"])
    train_unigram_vocab = record_unigram_vocabulary(train_df)
    unigram_vocab = list(set(train_unigram_vocab))
    filter_words_ls = ['gb', 'london', 'bexhill']
    unigram_vocab =  list(filter(lambda x: x not in filter_words_ls, unigram_vocab))
    tf = TfidfVectorizer(analyzer='word', 
        ngram_range=(1, 1), 
        vocabulary= train_unigram_vocab,
        stop_words='english')
        # token_pattern=r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)|\b\w\w+\b')
    tfidf_fit = tf.fit(train_df['Transaction Description'])
    tfidf_matrix_train = tfidf_fit.transform(train_df['Transaction Description'])
    tfidf_matrix_test = tfidf_fit.transform(test_df['Transaction Description'])
    cosine_similarities = linear_kernel(tfidf_matrix_train, tfidf_matrix_test)
    no_of_cols = cosine_similarities.shape[1]
    
    letter2_ls = record_2letter_words(unigram_vocab)
    char_vocab = list(set(letter2_ls + char_ngram(unigram_vocab,n =3)))
    tf_char = CountVectorizer(analyzer='char', ngram_range=(2, 3), vocabulary=char_vocab)
    tfidf_fit_char = tf_char.fit(train_df['Transaction Description'])
    tfidf_matrix_train_char = tfidf_fit_char.transform(train_df['Transaction Description'])
    tfidf_matrix_test_char = tfidf_fit_char.transform(test_df['Transaction Description'])
    cosine_similarities_char_ngram= linear_kernel(tfidf_matrix_train_char, tfidf_matrix_test_char)
    result = find_max_simialrity(cosine_similarities_char_ngram, cosine_similarities, no_of_cols, train_df, test_df)
    return SimilarityScoreCollection(similarity_score_collection = result)



