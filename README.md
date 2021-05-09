# deep-vaccine
Predict multi-epitope vaccine subunit candidates using NLP. Development setting for deep learning project. 
## Data
[Immune Epitope Database (IEDB)](https://www.iedb.org/)

- Randomly sample 40000 sequences from each file, and split into training and validation sets, so that there are no overlapping epitope fragments after combining T and B.
- When creating positive and negative datasets, the sequences are randomly shuffled to avoid the same pair of T and B being concatenated in T+B and B+T.
## Environment
[GCP VM](https://console.cloud.google.com/marketplace/product/click-to-deploy-images/deeplearning)
## Misc.
- Models: (CNN+) LSTM/GRU
- Different tokenizers and pooling
- Visualize models, data, training: `tensorboard --logdir=runs`