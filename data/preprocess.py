# %%
import os
import random
import pandas as pd
import numpy as np

# %%
random.seed(42)

# %%
class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# %%
Positive_T = DotDict()
Positive_B = DotDict()
Negative_T = DotDict()
Negative_B = DotDict()
Positive_train = DotDict()
Positive_test = DotDict()
Negative_train = DotDict()
Negative_test = DotDict()

# %%
data_dir = './data'
Positive_T.df = pd.read_csv(os.path.join(data_dir, 'Positive_T.csv'))
Positive_B.df = pd.read_csv(os.path.join(data_dir, 'Positive_B.csv'))
Negative_T.df = pd.read_csv(os.path.join(data_dir, 'Negative_T.csv'))
Negative_B.df = pd.read_csv(os.path.join(data_dir, 'Negative_B.csv'))
print(len(Positive_T.df))
print(len(Positive_B.df))
print(len(Negative_T.df))
print(len(Negative_B.df))

# %% [markdown]
# Randomly sample 40000 sequences from each file, and split into training and test sets, so that there are no overlapping fragments after combining T and B epitopes.

# %%
def sample(max_index, size=40000, split=8000):
    """Samples indices."""
    indices = random.sample(range(max_index), size)
    train = indices[:-split]
    test = indices[-split:]
    return train, test

# %%
Positive_T.train, Positive_T.test = sample(len(Positive_T.df))
Positive_B.train, Positive_B.test = sample(len(Positive_B.df))
Negative_T.train, Negative_T.test = sample(len(Negative_T.df))
Negative_B.train, Negative_B.test = sample(len(Negative_B.df))

# %% [markdown]
# Create positive and negative datasets. The sequences are randomly shuffled to avoid the same pair of B and T being concatenated in T+B and B+T.

# %%
def combine(df_T, df_B, indices_T, indices_B, outfile, outdir='./data'):
    list_TB = (df_T.iloc[indices_T,1].reset_index(drop=True) + df_B.iloc[indices_B,1].reset_index(drop=True)).to_list()
    list_BT = (df_B.iloc[random.sample(indices_B, len(indices_B)),1].reset_index(drop=True) + df_T.iloc[indices_T,1].reset_index(drop=True)).to_list()
    with open(os.path.join(outdir, outfile), 'w') as f:
        f.write('\n'.join(list_TB + list_BT))
    print(len(list_TB + list_BT))
    return list_TB + list_BT

# %%
Positive_train.lst = combine(Positive_T.df, Positive_B.df, Positive_T.train, Positive_B.train, 'Positive_train.txt')
Positive_test.lst = combine(Positive_T.df, Positive_B.df, Positive_T.test, Positive_B.test, 'Positive_test.txt')
Negative_train.lst = combine(Negative_T.df, Negative_B.df, Negative_T.train, Negative_B.train, 'Negative_train.txt')
Negative_test.lst = combine(Negative_T.df, Negative_B.df, Negative_T.test, Negative_B.test, 'Negative_test.txt')

