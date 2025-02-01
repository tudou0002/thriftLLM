import warnings
warnings.filterwarnings("ignore")

import random
import argparse

import MKnapsack
import AdaptSelect
import ThriftLLM

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="HEADLINES",help='Dataset to use.')
    parser.add_argument('--baseline', type=str, default="ThriftLLM", help='[Greedy, Octopus, ThriftLLM]')
    parser.add_argument('--eps', type=float, default=0.1, help='Parameter of approximation error.')
    parser.add_argument('--delta', type=float, default=0.01, help='Parameter of failure probability.')
    parser.add_argument('--B', type=float, default=0.0005, help='Total cost budget.')
    parser.add_argument('--thread', type=int, default=1, help='the number of threads')
    ### parameter of clustering ###
    parser.add_argument('--clsmethod', type=str, default="kmeans", help='[kmeans, cosine_dbscan]')
    parser.add_argument('--dis', type=float, default=0.5, help='distance in DBSCAN.')
    parser.add_argument('--minneighbor', type=int, default=10, help='minimum number of cluster size in DBSCAN')
    parser.add_argument('--ncluster', type=int, default=10, help='number of clusters in K-Means.')
    parser.add_argument('--PerClass', action="store_true", help='LLM Selection Per Query Class')
    parser.add_argument('--type', type=int, default=0, help='0/1 for text classification / entity matching')
    
    #####alabtion study##########
    parser.add_argument('--percent', type=float, default=1.0, help='distance in DBSCAN.')

    args = parser.parse_args()
    random.seed(42)

    print("--------------------------")
    print(args)

    ds_path_train = 'data/'+args.dataset+'/'+'Queried_'+args.dataset+'_all_models_clean_train.csv'
    ds_path_test = 'data/'+args.dataset+'/'+'Queried_'+args.dataset+'_all_models_clean_test.csv'
    emb_path = 'data/'+args.dataset+'/'+args.dataset+'_embeddings.json'

    
    myllm = ThriftLLM.ThriftLLM(ds_path_train, ds_path_test, emb_path, args.dataset, args.B, args.eps, args.delta, args.baseline,
                                args.clsmethod, args.dis, args.minneighbor, args.ncluster, args.percent, args.PerClass)
    
    t1 = time.time()
    #if args.PerClass: 
    if args.thread == 1:
        test_report = myllm.test(args.type)
    else: 
        test_report = myllm.test_parallel(args.type, thread_n=args.thread)       
        
    print('time: ', time.time() - t1)
    print(test_report)
