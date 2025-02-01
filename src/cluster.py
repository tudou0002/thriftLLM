import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class PreCluster:
    '''
    this class pre-clusters the samples, and returns the interested
    metrics for each cluster in a dict.
    '''

    def __init__(self, embeddings, gts, cls_mtd='kmeans'):
        self.embeddings = embeddings 
        self.cls_mtd = cls_mtd
        self.gts = gts 
        self.summary = {} 
        assert self.cls_mtd in ['kmeans', 'cosine_dbscan'], f'not supported clustering method: {self.cls_mtd}'

    def fit(self, n_cls=10, eps=0.1, min_samples=5, random_state = 191,):
        if self.cls_mtd == 'kmeans':
            kmeans = KMeans(n_clusters=n_cls, random_state=random_state, n_init='auto')
            kmeans.fit(self.embeddings)
            self.trained = kmeans
            cls_labels = kmeans.labels_ 
        elif self.cls_mtd == 'cosine_dbscan':
            # ensure all embeddings are already normalized
            # computes the cosine distance
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            dbscan.fit(self.embeddings)
            self.trained = dbscan # stores the trained objective
            cls_labels = dbscan.labels_ # cluster labels of all samples
            
        # -> report
        res = {} 
        cluster_num = self.trained.labels_.max() + 1
        for c_i in range(cluster_num):
            cls_mask = cls_labels == c_i
            cluster_gts = self.gts[cls_mask,:] # cls_size*m
            cluster_size = np.sum(cls_mask) # int
            gt_num_1 = np.sum(cluster_gts, axis=0) # (m,)
            gt_num_0 = cluster_size - gt_num_1 # (m,)
            # succ
            avg_succ1 = gt_num_1 / cluster_size # (m,)
            avg_succ = np.clip(avg_succ1, None, 1.0 - 1e-6)
            # std
            std = np.std(cluster_gts, axis=0)
            # consistency
            cons = np.maximum(gt_num_0/cluster_size, avg_succ) # (m,)
            # assembly
            cls_info = {'succ':avg_succ, 'std':std, 'cons':cons, 'n':cluster_size}
            res[c_i] = cls_info
        self.summary = res
        return res

    def assign(self, new_embed,):
        # new embedding --> nearest cluster index
        if self.cls_mtd == 'kmeans':
            centroids = self.trained.cluster_centers_
            distances = euclidean_distances(new_embed[None,:], centroids).flatten()
            return np.argmin(distances)
        elif self.cls_mtd == 'cosine_dbscan': # assign it to the nearest core sample
            core_sample_indices = self.trained.core_sample_indices_
            core_embeds = self.embeddings[core_sample_indices]
            core_labels = self.trained.labels_[core_sample_indices]
            distances = cosine_distances(new_embed[None,:], core_embeds).flatten()
            min_index = np.argmin(distances)
            return core_labels[min_index] # returns the cluster label of the cloest core sample
    
    def succ_p(self, new_embed,):
        # for a new embedding, predicts its succ rate in m models
        nn_c_i = self.assign(new_embed)
        return self.summary[nn_c_i]['succ']
