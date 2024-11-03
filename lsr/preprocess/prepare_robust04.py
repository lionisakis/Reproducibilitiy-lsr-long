import ir_datasets
from tqdm import tqdm
from pathlib import Path

print("Preparing msmarco document datasets")
robust04_dir = Path("data/trec-robust04")
dataset = ir_datasets.load("trec-robust04")

if not robust04_dir.is_dir():
    robust04_dir.mkdir(parents=True, exist_ok=True)

collection_path = robust04_dir / "collection.tsv"
with open(collection_path, "w", encoding="UTF-8") as f:
    for doc in tqdm(dataset.docs_iter()):
        text = doc.text.replace("\n", " ").replace("\t", " ")
        f.write(f"{doc.doc_id}\t{text}\n")

queries_path = robust04_dir / "desc-queries.tsv"
with open(queries_path, "w", encoding="UTF-8") as f:
    for query in tqdm(dataset.queries_iter()):
        desc = query.description.replace("\t", " ").replace("\n", " ")
        f.write(f"{query.query_id}\t{desc}\n")

qrels_path = robust04_dir / "robust04.qrels"
with open(qrels_path, "w", encoding="UTF-8") as f:
    for qrel in tqdm(dataset.qrels_iter()):
        f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
