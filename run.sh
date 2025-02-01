
for B in 0.00001 0.00005 0.0001 0.0005 0.001
do
    python3 main.py --thread 1  --B $B --dataset OVERRULING --clsmethod kmeans --ncluster 8 
done

for B in 0.00001 0.00005 0.0001 0.0005 0.001
do
    python3 main.py --thread 1 --clsmethod kmeans --B $B --dataset AGNEWS --ncluster 176 
done

for B in 0.00001 0.00005 0.0001 0.0005 0.001
do
    python3 main.py --thread 1 --clsmethod kmeans --B $B --dataset SCIQ --ncluster 1 
done

for B in 0.00001 0.00005 0.0001 0.0005 0.001
do
    python3 main.py --dataset HELLASWAG --B $B --thread 1 --clsmethod kmeans --ncluster 1  
done

for B in 0.00001 0.00005 0.0001 0.0005 0.001
do
    python3 main.py --thread 1 --clsmethod kmeans --B $B --dataset BANKING --ncluster 36 
done



for B in 0.00001 0.00005 0.0001 0.0005 0.001
do
    for dataset in ABTBUY DBLP WALMART WDC AMAZON SCHOLAR
    do 
        python3 main.py --dataset $dataset --B $B --thread 1 --clsmethod kmeans --ncluster 1 --type 1 >> new_$dataset\_thriftLLM_results
    done
done