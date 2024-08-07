This project is an extantion of [Adapting Learned Sparse Retrieval to Long Documents](https://github.com/thongnt99/lsr-long) repository

## Installation 
- Python packages
```console
conda create --name lsr python=3.9.12
conda activate lsr
pip install -r requirements.txt
```
- Python packages in a slurm environment
```console
sbatch slurm/scripts/gpu/install_environment.job
```
- Anserini for inverted indexing & retrieval:  Clone and compile [anserini-lsr](https://github.com/thongnt99/anserini-lsr), a customized version of Anserini for learned sparse retrieval. When compiling, add ```-Dmaven.test.skip=true``` to skip the tests.

## Downloading and spliting data 

* MSMARCO Documents
```console
bash scripts/prepare_msmarco_doc.sh
```
## BM25 baselines
```console
bash scripts/run_bm25_msmarco.sh 
```
## Simple aggregation 
To perform aggregation on MSMARCO, follow these steps. For TREC-Robust04, please modify the input and output files accordingly.
#### 1. Running inferences on segments (passages) and queries:
- segment inference (can be distributed on multiple gpus to speed up)
```console
sbatch slurm/scripts/gpu/inference_script_msmarcodoc_doc.job
```
- query inference
```console
sbatch slurm/scripts/gpu/inference_script_msmarcodoc_query.job
```
#### 2. Aggregating
- Representation max aggregation 
```console
sbatch slurm/scripts/cpu/aggregate_rep_msmarco_doc_max.job
```
- Representation mean aggregation
```console
sbatch slurm/scripts/cpu/aggregate_rep_msmarco_doc_mean.job
```
- Representation sum aggregation
```console
sbatch slurm/scripts/cpu/aggregate_rep_msmarco_doc_sum.job
```
- Score (max) aggregation
```console
sbatch slurm/scripts/cpu/aggregate_rep_msmarco_doc_max.job
```
## ExactSDM and SoftSDM

### ExactSDM 
| #Passages | MRR@10 | Script | 
|--------------|--------|---------|
| 1            |  37.00 | ```slurm/scripts/gpu/train_script_exact_sdm_long_reranker_1_psg.job``` |  
| 2            |  37.30 | ```slurm/scripts/gpu/train_script_exact_sdm_long_reranker_2_psg.sh``` |  
| 3            |  37.30 | ```slurm/scripts/gpu/train_script_exact_sdm_long_reranker_3_psg.sh``` |  
| 4            |  36.96 | ```slurm/scripts/gpu/train_script_exact_sdm_long_reranker_4_psg.sh``` |  
| 5            |  36.83 | ```slurm/scripts/gpu/train_script_exact_sdm_long_reranker_5_psg.sh``` |  

### SoftSDM

| #Passages | MRR@10 | Script | 
|--------------|--------|--------|
| 1            |  38.10 | ```slurm/scripts/gpu/train_script_sdm_long_reranker_1_psg.sh``` |  
| 2            |  37.41 | ```slurm/scripts/gpu/train_script_sdm_long_reranker_2_psg.sh``` |  
| 3            |  37.02 | ```slurm/scripts/gpu/train_script_sdm_long_reranker_3_psg.sh``` |  
| 4            |  36.64 | ```slurm/scripts/gpu/train_script_sdm_long_reranker_4_psg.sh``` |  
| 5            |  36.58 | ```slurm/scripts/gpu/train_script_sdm_long_reranker_5_psg.sh``` |  

## Global score and Global Injection for softSDM


### Global Score

| #Passages | MRR@10 | Script | 
|--------------|--------|--------|
| 1            |  32.32 | ```slurm/scripts/gpu/train_script_global_score_sdm_long_reranker_1_psg.job``` |  
| 2            |  31.65 | ```slurm/scripts/gpu/train_script_global_score_sdm_long_reranker_2_psg.job``` |  
| 3            |  30.83 | ```slurm/scripts/gpu/train_script_global_score_sdm_long_reranker_3_psg.job``` |  
| 4            |  30.71 | ```slurm/scripts/gpu/train_script_global_score_sdm_long_reranker_4_psg.job``` |  
| 5            |  31.11 | ```slurm/scripts/gpu/train_script_global_score_sdm_long_reranker_5_psg.job``` |  

### Global Injection 
| #Passages | MRR@10 | Script | 
|--------------|--------|--------|
| 1            |  33.18 | ```slurm/scripts/gpu/reranker_global_injected_qmlp_dmlm_msmarco_doc_1_psg.job``` |  
| 2            |  32.32 | ```slurm/scripts/gpu/reranker_global_injected_qmlp_dmlm_msmarco_doc_2_psg.job``` |  
| 3            |  32.14 | ```slurm/scripts/gpu/reranker_global_injected_qmlp_dmlm_msmarco_doc_3_psg.job``` |  
| 4            |  31.06 | ```slurm/scripts/gpu/reranker_global_injected_qmlp_dmlm_msmarco_doc_4_psg.job``` |  
| 5            |  32.35 | ```slurm/scripts/gpu/reranker_global_injected_qmlp_dmlm_msmarco_doc_5_psg.job``` |  

## Citing and Authors 
If you find this repository helpful, please check and cite the following papers:
- Adapting Learned Sparse Retrieval for Long Documents
```bibtex
@inproceedings{nguyen:sigir2023-llsr,
  author = {Nguyen, Thong and MacAvaney, Sean and Yates, Andrew},
  title = {Adapting Learned Sparse Retrieval for Long Documents},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2023}
}

```
- A Unified Framework for Learned Sparse Retrieval
```bibtex
@inproceedings{nguyen2023unified,
  title={A Unified Framework for Learned Sparse Retrieval},
  author={Nguyen, Thong and MacAvaney, Sean and Yates, Andrew},
  booktitle={Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2--6, 2023, Proceedings, Part III},
  pages={101--116},
  year={2023},
  organization={Springer}
}
```


