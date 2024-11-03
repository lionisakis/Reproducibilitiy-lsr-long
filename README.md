This project is a reproducibility of [Adapting Learned Sparse Retrieval to Long Documents](https://github.com/thongnt99/lsr-long) repository

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
bash slurm/scripts/cpu/prepare_msmarco_doc.sh
```
## BM25 baselines
```console
bash slurm/scripts/cpu/run_bm25_msmarco.sh 
```
## Simple aggregation 
To perform aggregation on MSMARCO, follow these steps. For TREC-Robust04, please modify the input and output files accordingly.
#### 1. Running inferences on segments (passages) and queries:
- segment inference (can be distributed on multiple gpus to speed up)
```console
sbatch slurm/scripts/gpu/inferences/inference_script_msmarcodoc_doc.job
```
- query inference
```console
sbatch slurm/scripts/gpu/inferences/inference_script_msmarcodoc_query.job
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

This section provides the Mean Reciprocal Rank (MRR@10) scores for different configurations of the ExactSDM and SoftSDM models. Each configuration varies by the number of passages, n-gram settings, and proximity values, with corresponding job scripts provided for reproducibility.

### ExactSDM
#### Estimating weights/Evaluating on MSMARCO Documents

| # Passages | Hyperparameters     | MRR@10 | Script |
|------------|----------------------|--------|--------|
| 1          | n-grams: 2, prox.: 8  | 36.99  | `slurm/scripts/gpu/train_exact/train_script_exact_sdm_long_reranker_1_psg_8_1_2.job` |
| 2          | n-grams: 2, prox.: 8  | 37.37  | `slurm/scripts/gpu/train_exact/train_script_exact_sdm_long_reranker_2_psg_8_1_2.sh` |
| 3          | n-grams: 2, prox.: 8  | 37.34  | `slurm/scripts/gpu/train_exact/train_script_exact_sdm_long_reranker_3_psg_8_1_2.sh` |
| 4          | n-grams: 2, prox.: 8  | 37.00  | `slurm/scripts/gpu/train_exact/train_script_exact_sdm_long_reranker_4_psg_8_1_2.sh` |
| 5          | n-grams: 2, prox.: 8  | 36.94  | `slurm/scripts/gpu/train_exact/train_script_exact_sdm_long_reranker_5_psg_8_1_2.sh` |
|---|---|---|---|
| 1          | n-grams: 5, prox.: 10 | 36.80  | `slurm/scripts/gpu/train_sdm/train_script_exact_long_reranker_1_psg_10_1_5.sh` |
| 2          | n-grams: 5, prox.: 10 | 36.51  | `slurm/scripts/gpu/train_sdm/train_script_exact_long_reranker_2_psg_10_1_5.sh` |
| 3          | n-grams: 5, prox.: 10 | 36.55  | `slurm/scripts/gpu/train_sdm/train_script_exact_long_reranker_3_psg_10_1_5.sh` |
| 4          | n-grams: 5, prox.: 10 | 36.42  | `slurm/scripts/gpu/train_sdm/train_script_exact_long_reranker_4_psg_10_1_5.sh` |
| 5          | n-grams: 5, prox.: 10 | 36.28  | `slurm/scripts/gpu/train_sdm/train_script_exact_long_reranker_5_psg_10_1_5.sh` |
#### Evaluating on TREC Robust04 (zero-shot)
```bash slurm/scripts/gpu/inferences/evaluate_original_exact_w_original_trec_robust.job```

### SoftSDM
#### Estimating weights/Evaluating on MSMARCO Documents
| # Passages | Hyperparameters     | MRR@10 | Script |
|------------|----------------------|--------|--------|
| 1          | n-grams: 2, prox.: 8  | 36.92  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_1_psg_8_1_2.sh` |
| 2          | n-grams: 2, prox.: 8  | 37.25  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_2_psg_8_1_2.sh` |
| 3          | n-grams: 2, prox.: 8  | 37.15  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_3_psg_8_1_2.sh` |
| 4          | n-grams: 2, prox.: 8  | 36.85  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_4_psg_8_1_2.sh` |
| 5          | n-grams: 2, prox.: 8  | 36.80  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_5_psg_8_1_2.sh` |
|---|---|---|---|
| 1          | n-grams: 5, prox.: 10 | 37.01  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_1_psg_10_1_5.sh` |
| 2          | n-grams: 5, prox.: 10 | 37.27  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_2_psg_10_1_5.sh` |
| 3          | n-grams: 5, prox.: 10 | 37.07  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_3_psg_10_1_5.sh` |
| 4          | n-grams: 5, prox.: 10 | 36.76  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_4_psg_10_1_5.sh` |
| 5          | n-grams: 5, prox.: 10 | 36.65  | `slurm/scripts/gpu/train_sdm/train_script_sdm_long_reranker_5_psg_10_1_5.sh` |
#### Evaluating on TREC Robust04 (zero-shot)
```bash slurm/scripts/gpu/inferences/evaluate_original_soft_w_original_trec_robust.job```

## Impact Analysis

### Relevance Variation of Document Segments
To conduct this analysis, use the Jupyter notebook `analysis_psg.ipynb`, which allows you to assess multiple documents as needed. As this can be time-consuming, an alternative Python script, `analysis_psg.py`, is also available. This script can be executed on a Slurm server by running the job `slurm/scripts/cpu/analysis_psg.job`.

### Dependence on Sparse Segment Representation
For this analysis, execute the Jupyter notebook `vector_segmantics.ipynb`.

### Effect of Segment Count and Document Length on Document-Query Scoring

To analyze how segment count and document length impact scoring methods, follow these steps:

1. Download the MS-MARCO Document dataset.
2. Run the Jupyter notebook `vector_segmantics.ipynb` to create shorter versions of the documents (e.g., short and long).
3. Choose an evaluation method:
   - To evaluate **softSDM**, submit the job `slurm/scripts/gpu/inferences/evaluate_original_sofr_w_altered_msmarco.job`.
   - For **ExactSDM**, use `slurm/scripts/gpu/inferences/evaluate_original_exact_w_altered_msmarco.job`.

<!-- ## Citing and Authors 
If you find this repository helpful, please check and cite the original papers:
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

 -->
