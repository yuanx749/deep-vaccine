# deep-vaccine
Predict multi-epitope vaccine subunit candidates using NLP.

## Data
[Immune Epitope Database (IEDB)](https://www.iedb.org/)

Construct datasets with `data/preprocess.py` (notebook format used by mainstream editors).

## Environment
- [Google Cloud Platform VM](https://console.cloud.google.com/marketplace/product/click-to-deploy-images/deeplearning)
- [Google Colab](https://colab.research.google.com/)

## Usage
To train, `python train.py`, or use `LSTM.ipynb` on Colab. Models are saved in `./runs`.

Predict a list of sequences using a model saved at `path_to_model` as follows:
```python
from api import Predictor

seqs = """
PVAGAAIAAPVAGQQGPQRR
IAADFVEDQEVCKNYTGTVVGFASMVA
ADGAYRFLSGTAAVLAAAETAEAKAAAAAE
GDNLKGIVVIKDRNIGVLGENGSHMPDRCN
""".split()

predictor = Predictor(path_to_model)
predictor.predict_proba(seqs)
```

An application on the spike protein of SARS-CoV-2 is in `example.py`.

## Misc.
- Models: (CNN+) LSTM/GRU, Transformer
- Different tokenizers and pooling
- Visualize models, data, training: `tensorboard --logdir=runs`

If you find this helpful, please consider citing ([bib](CITATION.bib)).
