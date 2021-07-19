# deep-vaccine
Predict multi-epitope vaccine subunit candidates using NLP.
## Data
[Immune Epitope Database (IEDB)](https://www.iedb.org/)

Construct datasets: `python ./data/preprocess.py`
## Environment
[GCP VM](https://console.cloud.google.com/marketplace/product/click-to-deploy-images/deeplearning)
## Misc.
- Models: (CNN+) LSTM/GRU, Transformer
- Different tokenizers and pooling
- Visualize models, data, training: `tensorboard --logdir=runs`