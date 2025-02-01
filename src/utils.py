from tqdm import tqdm
import re

import numpy as np
from sklearn.neighbors import NearestNeighbors
from nltk.translate.bleu_score import sentence_bleu

# headlines
def make_query_raw(row):
    row['query_raw'] = [e for e in row['query_raw'].split('\n') if e][-2]
    return row

def sentence_decode(sentence):
    pattern = r"\$?\d+(?:,\d+)*(?:\.\d+)?%?|\b[A-Za-z]+(?:'[A-Za-z]+)?\b"
    matches = re.findall(pattern, sentence)
    return matches

def decode_row(row, models, answers_set):
    q_raw = row['query_raw']
    ref_ans = row['ref_answer']
    if type(ref_ans) == str:
        ref_ans = row['ref_answer'].replace('\n', '').replace(' ', '')
    pred_ans = [str(row[x]).replace('\n', '').replace(' ', '') for x in models]
    assert ref_ans in answers_set 
    return q_raw, ref_ans, pred_ans


def headlines_emb_key(q_raw):
    return q_raw.split('\n')[0].replace('Q:', '')

def overruling_emb_key(q_raw):
    return q_raw.split('\n')[0].replace('Context:', '')

def agnews_emb_key(q_raw):
    q_key = q_raw.split('\n')[0].split('Q:')[1]
    return q_key

def hellaswag_emb_key(q_raw):
    q_raw1 = q_raw.replace('Context:', '')
    q_split = q_raw1.split('\n')
    q_key = q_split[0] + q_split[4].replace('3. ', '-1. ', 1)
    return q_key 

def sciq_emb_key(q_raw):
    return q_raw.replace('\n', ' ')

def banking_emb_key(q_raw):
    q_key = q_raw.split('\n')[0].split('Query: ')[1]
    if q_key == '':
        q_key = q_raw.split('\n')[1]
    return q_key

def entityembedding_emb_key(q_raw):
    q_key = q_raw.split('\n')

    l = 4
    if 'Entity' == q_key[0][:6]: l += 6
    if 'Product' == q_key[0][:7]: l += 7
    if 'Publication' == q_key[0][:11]: l += 11
    
    q_key = q_key[0][l:] + ' ' + q_key[1][l:]
    return q_key

###############################################

# similarity search
class SimilaritySearch:
    def __init__(self, X, K=5, metric='cosine'):
        self.X = X
        self.K = K
        assert metric in ['cosine', 'euclidean', 'bleu']
        self.metric = metric
        self.skl_metric_map = {'cosine': 'cosine',
                               'euclidean': 'minkowski',}
        if metric in self.skl_metric_map.keys():
            self.skl_nnbrs = NearestNeighbors(n_neighbors=K,
                                            metric=self.skl_metric_map[metric],
                                            p=2)
            self.skl_nnbrs.fit(X)
        
    def bleu_kneighbors(self, Y):
        # -> nn_test_scores, nn_test_indices
        Y_clean = Y
        X_clean = self.X
        # we assume the sentences are cleaned by sentence_decode
        scores_matrix = np.zeros([len(Y_clean), len(X_clean)])
        for i in tqdm(range(len(Y_clean)), total=len(Y_clean)):
            refs = [Y_clean[i]]
            for j in range(len(X_clean)):
                hypo = X_clean[j]
                scores_matrix[i,j] = sentence_bleu(refs, hypo)
        
        sorted_indices = np.argsort(scores_matrix, axis=1)[:, ::-1]
        top_k_indices = sorted_indices[:, :self.K] # (Y, K)
        
        nn_test_scores = scores_matrix[np.arange(len(Y_clean)).reshape((-1,1)), top_k_indices] # (Y,K)
        return nn_test_scores, top_k_indices

    def kneighbors(self, Y):
        if self.metric in self.skl_metric_map.keys():
            return self.skl_nnbrs.kneighbors(Y)
        elif self.metric == 'bleu':
            return self.bleu_kneighbors(Y)

# data processing
def norm_rows(X):
    # X: np.array
    rows_norm = np.linalg.norm(np.array(X), ord=2, axis=1, keepdims=True)
    return X/rows_norm

def weighted_norm(X,Y,ratio=1):
    # assume X and Y are normlized already
    # ratio: how important is Y compared to X, by default is 1 (means equally important)
    combined = np.concatenate((np.array(X), np.array(Y)*ratio**0.5), axis=1)
    return norm_rows(combined)
