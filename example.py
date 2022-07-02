# %%
import pandas as pd

from api import Predictor, read_fasta, extract_kmers

# %%
protein = read_fasta("P0DTC2.fasta")[0]
path_to_model = ""  # replace with the real value
model = Predictor(path_to_model)

# %%
k = 30
starts = range(len(protein) - k + 1)
kmers = extract_kmers(protein, k)
probs = model.predict_proba(kmers)
df = pd.DataFrame({"Start": starts, "Sequence": kmers, "Probability": probs})

# %%
threshold = 0.999
with pd.option_context("display.max_rows", None):
    subunits = df[df["Probability"] > threshold].reset_index(drop=True)
    print(subunits)
