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

# %%
def z_description(aa):
    z1, z2, z3 = 0, 0, 0
    if aa == "A":
        z1 = 0.07
        z2 = -1.73
        z3 = 0.09
    elif aa == "V":
        z1 = -2.69
        z2 = -2.53
        z3 = -1.29
    elif aa == "L":
        z1 = -4.19
        z2 = -1.03
        z3 = -0.98
    elif aa == "I":
        z1 = -4.44
        z2 = -1.68
        z3 = -1.03
    elif aa == "P":
        z1 = -1.22
        z2 = 0.88
        z3 = 2.23
    elif aa == "F":
        z1 = -4.92
        z2 = 1.30
        z3 = 0.45
    elif aa == "W":
        z1 = -4.75
        z2 = 3.65
        z3 = 0.85
    elif aa == "M":
        z1 = -2.49
        z2 = -0.27
        z3 = -0.41
    elif aa == "K":
        z1 = 2.84
        z2 = 1.41
        z3 = -3.14
    elif aa == "R":
        z1 = 2.88
        z2 = 2.52
        z3 = -3.44
    elif aa == "H":
        z1 = 2.41
        z2 = 1.74
        z3 = 1.11
    elif aa == "G":
        z1 = 2.23
        z2 = -5.36
        z3 = 0.30
    elif aa == "S":
        z1 = 1.96
        z2 = -1.63
        z3 = 0.57
    elif aa == "T":
        z1 = 0.92
        z2 = -2.09
        z3 = -1.40
    elif aa == "C":
        z1 = 0.71
        z2 = -0.97
        z3 = 4.13
    elif aa == "Y":
        z1 = -1.39
        z2 = 2.32
        z3 = 0.01
    elif aa == "N":
        z1 = 3.22
        z2 = 1.45
        z3 = 0.84
    elif aa == "Q":
        z1 = 2.18
        z2 = 0.53
        z3 = -1.14
    elif aa == "D":
        z1 = 3.64
        z2 = 1.13
        z3 = 2.36
    elif aa == "E":
        z1 = 3.08
        z2 = 0.39
        z3 = -0.07
    return z1, z2, z3

# %%
def acc(seq):
    n = len(seq)
    accn = np.zeros(45)
    z = np.array([z_description(aa) for aa in seq])
    z_product = np.array([
        z[:-1,0] * z[1:,0],
        z[:-1,1] * z[1:,1],
        z[:-1,2] * z[1:,2],
        z[:-1,0] * z[1:,1],
        z[:-1,0] * z[1:,2],
        z[:-1,1] * z[1:,0],
        z[:-1,1] * z[1:,2],
        z[:-1,2] * z[1:,0],
        z[:-1,2] * z[1:,1]
    ]).T

    for l in range(1, 6):
        accn[9*(l-1):9*l] = np.mean(z_product[:n-l], axis=0)

    return accn

# %%
np.save(os.path.join(data_dir, 'Positive_train.npy'), np.array([acc(seq) for seq in Positive_train.lst]))
np.save(os.path.join(data_dir, 'Positive_test.npy'), np.array([acc(seq) for seq in Positive_test.lst]))
np.save(os.path.join(data_dir, 'Negative_train.npy'), np.array([acc(seq) for seq in Negative_train.lst]))
np.save(os.path.join(data_dir, 'Negative_test.npy'), np.array([acc(seq) for seq in Negative_test.lst]))

# %%
