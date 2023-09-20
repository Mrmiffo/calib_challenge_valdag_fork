import torch
import os
from sklearn import linear_model
import numpy as np
import sys

EMBEDDINGS_DIR = "raft_embeddings"
LABELS_DIR = "labeled"
UNLABELED_DIR = "unlabeled"

TEXT_FILE_EXTESION = ".txt"


if len(sys.argv) > 1:
  TEST_DIR = sys.argv[1]
else:
  raise RuntimeError('No test directory provided')

os.makedirs(TEST_DIR, exist_ok=True)

reg = linear_model.LinearRegression()

all_embeddings = None
all_labels = None

label_files = filter(lambda file: os.path.splitext(file)[1] == TEXT_FILE_EXTESION, os.listdir(LABELS_DIR))
for label_file in label_files:
    labels_path = os.path.join(LABELS_DIR, label_file)

    file_name = os.path.splitext(label_file)[0]
    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{file_name}.pt")
    assert os.path.isfile(embeddings_path), f"Missing embeddings, {embeddings_path}"

    embeddings = torch.load(embeddings_path)

    if all_embeddings is None:
        all_embeddings = embeddings
    else:
        all_embeddings = np.append(all_embeddings, embeddings, 0)

    labels = np.loadtxt(labels_path)
    # Skip the last label since it doesn't have a corresponding embedding
    labels = labels[:-1]

    if all_labels is None:
        all_labels = labels
    else:
        all_labels = np.append(all_labels, labels, 0)

not_nan_indices = ~np.isnan(all_labels).any(axis=1)

all_embeddings = all_embeddings[not_nan_indices,:]
all_labels = all_labels[not_nan_indices,:]

# Fit the model
reg.fit(all_embeddings, all_labels)

# Make predictions
for unlabeled_video in os.listdir(UNLABELED_DIR):

    unlabeled_name = os.path.splitext(unlabeled_video)[0]
    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{unlabeled_name}.pt")
    assert os.path.isfile(embeddings_path), f"Missing embeddings, {embeddings_path}"

    embeddings = torch.load(embeddings_path)

    predictions = reg.predict(embeddings)
    # Set the last prediction to nan since it doesn't have a corresponding embedding
    predictions = np.append(predictions, np.array([[np.nan, np.nan]]), axis=0)

    with open(os.path.join(TEST_DIR, f"{unlabeled_name}.txt"), "w") as f:
        np.savetxt(f, predictions)
