# Data

**UNSW-NB15** (UNSW Canberra Cyber), the official partition files:

| File | Rows | SHA-256 |
|---|---|---|
| `UNSW_NB15_training-set.csv` | 175,341 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| `UNSW_NB15_testing-set.csv` | 82,332 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |

The CSVs are **not** in this repository. `advml.data.ensure_dataset()` downloads them from the Hugging Face mirror
`Mireu-Lab/UNSW-NB15`, pinned to commit `3eb02c4a`, and refuses any file whose hash differs. The mirror names the two files
the other way round (`train.csv` is the official testing set); the code maps them back by hash and row count.

Use:
- all tuning on a validation split carved (stratified, 80/20, seed 1337) from the training set;
- the testing set is evaluated **once**, at the end (`python -m advml final`, which refuses to run a second time).

Licence and citation: free for academic research use; cite
N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network
data set)", Military Communications and Information Systems Conference (MilCIS), 2015.
