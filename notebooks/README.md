# Notebooks

This folder contains the main Jupyter notebooks used for the experimental evaluation of graph embedding methods in the project **Evaluation of Graph Embedding Methods on Real-World Datasets**.

Each notebook corresponds to a core task defined in the project specification:

## 1. `classification.ipynb`
Implements the **graph classification** pipeline.
- Computes graph embeddings (e.g., Graph2Vec, NetLSD, GIN)
- Trains a downstream classifier (SVM)
- Reports classification metrics (accuracy, F1-score, AUC)
- Records embedding and training time

## 2. `clustering.ipynb`
Implements **unsupervised clustering** on graph embeddings.
- Applies k-means / spectral clustering
- Reports clustering quality metrics (e.g., ARI)
- Produces qualitative visualizations (t-SNE / UMAP)

## 3. `stability_new.ipynb`
Implements the **stability analysis** of graph embeddings.
- Introduces controlled graph perturbations (edge noise, attribute shuffling)
- Recomputes embeddings
- Measures embedding drift and changes in downstream performance

Together, these notebooks cover the classification, clustering, and stability evaluation tasks required for the project.