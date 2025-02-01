import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
import numpy as np

import utils
from parameters import *
import cluster
import price
import AdaptSelect as adapt

###### import functions #############
from AdaptSelect import *
import math

import time



class ThriftDataset:
    '''
    Given a frugal-format dataset, this data structure constructs a 3-d tensor with
    following dimentions: query, model and answer, which can be accessed through
    thrift_ds.data.
    '''
    def __init__(self, ds_path, dataset, percent):
        self.df = pd.read_csv(ds_path,keep_default_na=False).astype(str)
        self.df = self.df[:int(len(self.df) * percent)]
        self.DATASET = dataset
        # stats on models
        self.models = self.df.columns.to_list()[3:]   # to be checked for all datasets
        self.model2idx = {m:i for i,m in enumerate(self.models)}
        self.L = len(self.models)
        # stats on classes
        self.classes = sorted(self.df['ref_answer'].unique().tolist())
        self.class2idx = {c:i for i,c in enumerate(self.classes)}
        self.K = len(self.classes)
        # stats on queries
        self.queries = sorted(self.df['query_raw'].unique().tolist())
        self.query2idx = {q:i for i,q in enumerate(self.queries)}
        self.N = len(self.queries)
        # transfer the data into a more condense form
        self.data = np.full((self.N, self.L, 2), -1) # |queries| * |models| * |(gt_idx, answer_idx)|
        self.fill_the_data() # construct the self.data 3-d tensor
    
    def fill_the_data(self,):
        if self.DATASET not in {'HEADLINES','OVERRULING','AGNEWS','HELLASWAG','SCIQ','BANKING','ABTBUY','DBLP','WALMART','WDC','AMAZON','SCHOLAR'}:
            assert False, f'dataset {self.DATASET} not supported yet'
        for _,row in self.df.iterrows():
            q_raw, ref_ans, pred_anses = utils.decode_row(row, self.models, self.classes) # q_key: to find the embedding
            q_i = self.query2idx[q_raw]
            for model_idx in range(self.L):
                model_name = self.models[model_idx]
                pred_ans = pred_anses[model_idx]
                m_i = self.model2idx[model_name]
                self.data[q_i, m_i, 0] = self.class2idx[ref_ans]
                if pred_ans not in self.classes: 
                    self.data[q_i, m_i, 1] = (self.data[q_i, m_i, 0] + 1) % self.K
                    continue               
                self.data[q_i, m_i, 1] = self.class2idx[pred_ans]
        neg_count = np.sum(self.data == -1)
        if neg_count: print(f'warning: number of cell unfilled: {neg_count}')

class ThriftLLM:
    # The main class of the ThriftLLM, should be initialized with a dataset and embeddings.
    # For a new query q in the test dataset, the user should give the corresponding embedding.
    def __init__(self, ds_path_train, ds_path_test, emb_path, dataset, B, eps, delta, baseline, clsmethod, dis, minneighbor, ncluster, percent, PerClass):
        # load dataset (train, test) and embeddings
        # the embedding data should include both train and test ds queries
        self.DATASET = dataset
        print(f'loading embeddings...')
        with open(emb_path, 'r') as f: # .json file, should include embeddings for all train/test samples
            self.ds_embeddings = json.load(f)
        print(f'loading train and test dataset...')
        self.thrift_ds_train = ThriftDataset(ds_path_train, dataset, percent)
        self.thrift_ds_test = ThriftDataset(ds_path_test, dataset, 1.0)
        self.answer_chache = {} # cache: tuple -> llm subset
        self.eps = eps
        self.delta = delta
        self.B = B
        self.baseline = baseline
        self.PerClass = PerClass
        # pre-clusters
        # we need to cluster according to the train set for each of the model
        print(f'clustering...')
        self.prepare_clusters(clsmethod, dis, minneighbor, ncluster)
        print(f'initialization on {self.DATASET} is done.')

    def make_query_key(self, q_raw):
        if self.DATASET == 'HEADLINES':
            q_key = utils.headlines_emb_key(q_raw)
        elif self.DATASET == 'OVERRULING':
            q_key = utils.overruling_emb_key(q_raw)
        elif self.DATASET == 'AGNEWS':
            q_key = utils.agnews_emb_key(q_raw)
        elif self.DATASET == 'HELLASWAG':
            q_key = utils.hellaswag_emb_key(q_raw)
        elif self.DATASET == 'SCIQ':
            q_key = utils.sciq_emb_key(q_raw)
        elif self.DATASET == 'BANKING':
            q_key = utils.banking_emb_key(q_raw)
        elif self.DATASET in {'ABTBUY','DBLP','WALMART','WDC','AMAZON','SCHOLAR'}:
            q_key = utils.entityembedding_emb_key(q_raw)
        else:
            assert False, f'dataset {self.DATASET} not supported yet' # TODO: expand
        return q_key

    def prepare_clusters(self,clsmethod, dis, minneighbor, ncluster):
        # make common embeddings list
        common_embeddings = []
        for q_raw in self.thrift_ds_train.queries:
            q_key = self.make_query_key(q_raw)
            common_embeddings.append(self.ds_embeddings[q_key])
        common_embeddings = np.array(common_embeddings) # embeddings for all training queries in order

        # make pre-cluster for each model
        dsdata = self.thrift_ds_train.data
        gts = dsdata[:, :, 0] == dsdata[:, :, 1] # (n, m)
        self.pre_cluster_obj = cluster.PreCluster(common_embeddings, gts, cls_mtd=clsmethod)
        self.cls_res = self.pre_cluster_obj.fit(ncluster, dis, minneighbor, CLS_RANDOM_STATE)

        self.T_cache = [[]] * len(self.cls_res)

    def succ_p_all(self, q, emb,):
        return self.pre_cluster_obj.succ_p(emb,)

    def answer_class(self, q, emb):
        costs = price.price_all(q, self.thrift_ds_train.models) 
        probs = self.succ_p_all(q, np.array(emb)) 

        cls = 0
        if len(self.T_cache[cls]) > 0:
            S, pre = adapt.AdaptSelect_prefix(q, probs, self.thrift_ds_train.K,
                                            self.thrift_ds_test, self.T_cache[cls])
            return pre, sum([costs[x] for x in S])
        
        T, S, pre = adapt.AdaptSelect(q, probs, costs, self.B, self.eps, self.delta,
                                          self.thrift_ds_train.K, self.thrift_ds_test, self.thrift_ds_train.L)
        self.T_cache[cls] = self.T_cache[cls] + T
        return pre, sum([costs[x] for x in S])

    def answer(self, q, emb):
        # let ThriftLLM answer an in-test query -> pred (class index)
        # computes the expected costs for all models for this query
        costs = price.price_all(q, self.thrift_ds_train.models) # [m]
        # measures the success rate for all models for this query
        probs = self.succ_p_all(q, np.array(emb))  # np.array
        _, pst = pstar(probs, costs, self.B)
        theta = round((8 + 2 * self.eps) / self.eps / self.eps / pst * math.log(2 * len(probs) * len(probs) / self.delta)) + 1

        if self.baseline == 'Greedy':
            S = Greedy(probs, costs, self.B, self.thrift_ds_train.K, theta)
            return Prediction(S, probs, q, self.thrift_ds_test, self.thrift_ds_train.K), sum([costs[x] for x in S]), self.B
        if self.baseline == 'Octopus':
            S = Octopus(probs, costs, self.B)
            return Prediction(S, probs, q, self.thrift_ds_test, self.thrift_ds_train.K), sum([costs[x] for x in S]), self.B
        
        T = Greedy(probs, costs, self.B, self.thrift_ds_train.K, theta)
        T = SanGreedy(probs, costs, self.B, self.thrift_ds_train.K, theta)
        T, S, pre = adapt.AdaptSelect(T, q, probs, self.thrift_ds_train.K, self.thrift_ds_test)
        
        return pre, sum([costs[x] for x in S]), sum([costs[x] for x in T])

    def test(self, type=0, debug_num=None):
        '''
        Answer all the queries in the test set

        debug_num: number of samples to test
        '''
        print('warning: plz use the test_parallel')
        test_report = {}
        correct_count = 0.0
        pos_cnt = 0.0
        pos_correct_cnt = 0.0
        task_queries = self.thrift_ds_test.queries
        if debug_num is not None:
            task_queries = task_queries[:debug_num]

        total_cost = 0.0
        total_exp_cost = 0.0
        for q in tqdm(task_queries):
            qk = self.make_query_key(q)
            emb = self.ds_embeddings[qk]
            if self.PerClass: pred, real_costs = self.answer_class(q, emb)
            else: pred, real_costs, exp_costs = self.answer(q, emb)
            q_i = self.thrift_ds_test.query2idx[q]
            ref = self.thrift_ds_test.data[q_i][0][0]
            if ref == pred: correct_count += 1
            if type == 1 and ref == self.thrift_ds_test.class2idx['yes']: 
                pos_cnt += 1
                pos_correct_cnt += (ref == pred)

            total_cost += real_costs
            total_exp_cost += exp_costs
        test_report['acc'] = correct_count / len(task_queries)
        test_report['total-cost'] = total_cost
        test_report['exp-cost'] = total_exp_cost
        test_report['avg-cost'] = total_cost / len(task_queries)
        test_report['avg-exp-cost'] = total_exp_cost / len(task_queries)
        if type == 1:
            test_report['recal'] = pos_correct_cnt / pos_cnt
            test_report['F1-score'] = 2.0 * test_report['acc'] * test_report['recal'] / (test_report['acc'] + test_report['recal'])

        return test_report 
    
    # single thread task
    def thread_task(self, q):
        qk = self.make_query_key(q)
        emb = self.ds_embeddings[qk]
        if self.PerClass: pred, real_costs = self.answer_class(q, emb)
        else: pred, real_costs, exp_costs = self.answer(q, emb)
        q_i = self.thrift_ds_test.query2idx[q]
        ref = self.thrift_ds_test.data[q_i][0][0]
        return ref == pred, real_costs, exp_costs, ref

    def test_parallel(self, type=0, thread_n=12, debug_num=None):
        test_report = {}
        correct_count = 0
        total_cost = 0.0 
        total_exp_cost = 0.0
        pos_cnt = 0.0
        pos_correct_cnt = 0.0
        task_queries = self.thrift_ds_test.queries
        if debug_num is not None: # selects a subset
            task_queries = task_queries[:debug_num]
        with tqdm(total=len(task_queries)) as pbar:
            with ProcessPoolExecutor(max_workers=thread_n) as executor:
                futures = [executor.submit(self.thread_task, q) for q in task_queries]
                for future in as_completed(futures):
                    pbar.update(1)
                    if_correct, real_cost, exp_cost, ref = future.result()
                    if type == 1 and ref == self.thrift_ds_test.class2idx['yes']: 
                        pos_cnt += 1
                        pos_correct_cnt += if_correct
                    correct_count += if_correct
                    total_cost += real_cost
                    total_exp_cost += exp_cost
        test_report['acc'] = correct_count / len(task_queries)
        test_report['total-cost'] = total_cost
        test_report['exp-cost'] = total_exp_cost
        test_report['avg-cost'] = total_cost / len(task_queries)
        test_report['avg-exp-cost'] = total_exp_cost / len(task_queries)
        if type == 1:
            test_report['recal'] = pos_correct_cnt / pos_cnt
            test_report['F1-score'] = 2.0 * test_report['acc'] * test_report['recal'] / (test_report['acc'] + test_report['recal'])
        return test_report
    
 



