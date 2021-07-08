from SimariltyResult import SimilarityScore, SimilarityScoreCollection
import numpy as np


def normalize_count_cosine(arr):
    norm = np.linalg.norm(arr)
    return arr/norm


def find_max_simialrity(cosine_similarities_char_ngram,
                        cosine_similarities,
                        no_of_cols, train_df, test_df):
    max_cos_ls = []
    for i in range(no_of_cols):
        normed_arr = normalize_count_cosine(
            cosine_similarities_char_ngram[:, i])
        cosine_similariteis_av = (normed_arr + cosine_similarities[:, i])/2
        max_cos_idx = np.argmax(cosine_similariteis_av)
        max_cos = max(cosine_similariteis_av)
        if max_cos > 0.53:
            ss= SimilarityScore(similarity_score=max_cos,
                            similarity_score_index=max_cos_idx,
                            test_transaction_description=test_df['Transaction Description'][i],
                            train_transaction_description=train_df['Transaction Description'][max_cos_idx],
                            predicted_submerchant=train_df['SubMerchant'][max_cos_idx])

            max_cos_ls.append(
                (ss))
    return max_cos_ls
