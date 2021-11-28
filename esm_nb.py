# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import esm
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

# %%
from data import (
    AntigenDataset,
    BaselineDataset,
    EpitopeDataset,
    EpitopeRawDataset,
    Tokenizer,
    Tokenizer2,
)
from utils import plot_roc_curve

# %%
pretrained_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

# %%
RANDOM_SEED = 42


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# %%
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer(max_len=40)
train_dataset = EpitopeRawDataset(
    "Positive_train.txt", "Negative_train.txt", tokenizer=tokenizer, data_dir="./data"
)
test_dataset = EpitopeRawDataset(
    "Positive_test.txt", "Negative_test.txt", tokenizer=tokenizer, data_dir="./data"
)

# %%
def get_representations(dataset, pretrained_model, alphabet, batch_size=128):
    """Returns: N x 1280 numpy array"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    batch_converter = alphabet.get_batch_converter()

    sequence_representations = []
    progress_bar = tqdm(dataloader, ascii=True)
    for i, (tokens, labels, seqs) in enumerate(progress_bar):
        esm_batch = list(zip(labels, seqs))
        batch_labels, batch_strs, batch_tokens = batch_converter(esm_batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = pretrained_model(
                batch_tokens, repr_layers=[33], return_contacts=True
            )
        token_representations = results["representations"][33]
        outputs = token_representations[:, 0]  # get the <cls> token
        sequence_representations.append(outputs.cpu().numpy())
    return np.vstack(sequence_representations)


# %%
train_representations = get_representations(train_dataset, pretrained_model, alphabet)
torch.save(train_representations, "./data/train_esm.pkl")

# %%
test_representations = get_representations(test_dataset, pretrained_model, alphabet)
torch.save(test_representations, "./data/test_esm.pkl")

# %%
train_representations = torch.load("./data/train_esm.pkl")
test_representations = torch.load("./data/test_esm.pkl")

# %%
lr = LogisticRegression(max_iter=2000)
lr.fit(train_representations, train_dataset.labels)
lr.score(train_representations, train_dataset.labels)

# %%
lr.score(test_representations, test_dataset.labels)

# %%
figure = plot_roc_curve(
    train_dataset.labels,
    lr.predict_proba(train_representations)[:, 1],
    test_dataset.labels,
    lr.predict_proba(test_representations)[:, 1],
)
