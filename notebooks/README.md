# classification.ipynb: Graph Classification Pipelines (GIN, Graph2Vec, NetLSD)

This notebook implements three graph classification pipelines on **TUDataset** benchmarks:

- **GIN (Graph Isomorphism Network)** using PyTorch Geometric  
- **Graph2Vec + SVM** using KarateClub  
- **NetLSD + SVM** using KarateClub  

It also includes **Optuna hyperparameter tuning**, **metrics evaluation**, and **logging** of results.

---

## 1. Dataset Handling

- Uses `torch_geometric.datasets.TUDataset`  
- If node features are missing, applies `OneHotDegree`  
- Converts PyG graphs → NetworkX graphs for Graph2Vec & NetLSD  
- Includes **special handling for ENZYMES** (filters graphs with very few nodes)  
- Embeddings are sanitized to remove `NaN` / `Inf` values

---

## 2. GIN Pipeline

**Function:** `run_gin_pipeline()`  
- 80/20 train-test split  
- Optional **Optuna tuning** (layers, dropout, LR, weight decay)  
- Tracks accuracy, F1, AUC, loss  
- Logs results in `results/training_log.csv`  
- Saves model in `models/GIN_<dataset>.pth`
- Saves embeddings/labels in `embeddings/` for future use 

---

## 3. Graph2Vec Pipeline

**Function:** `run_graph2vec_pipeline()`  
- Converts graphs to NetworkX  
- Optional Optuna tuning (SVM `C`, `gamma`)  
- Transductive embedding: fits Graph2Vec on train+val or train+test  
- Evaluates accuracy, F1, AUC, combined weighted score  
- Logs results in `results/graph2vec_log.csv`
- Saves embeddings/labels in `embeddings/` for future use

---

## 4. NetLSD Pipeline

**Function:** `run_netlsd_pipeline()`  
- Similar workflow to Graph2Vec  
- Uses NetLSD spectral signatures  
- Optional Optuna tuning for SVM parameters  
- Logs results in `results/netlsd_log.csv`
- Saves embeddings/labels in `embeddings/` for future use

---

## 5. Metrics & Logging

Each pipeline reports:
- **Accuracy**
- **Weighted F1**
- **AUC (binary or multiclass)**
- **Weighted combined score** using (w_acc, w_f1, w_auc)
- **Embedding time**, **training time**, **Optuna time**
- **Memory usage**

All results are appended to CSV log files for experiment tracking.

---

## 6. Example Usage

```python
run_gin_pipeline("MUTAG", use_optuna=True, w_acc=0.5, w_f1=0.3, w_auc=0.2, hidden_dim=64, epochs=50)
run_graph2vec_pipeline("MUTAG", use_optuna=True, n_trials=20)
run_netlsd_pipeline("MUTAG", use_optuna=True, n_trials=20)