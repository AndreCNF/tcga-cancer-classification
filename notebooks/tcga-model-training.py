# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TCGA Model training
# ---
#
# Experimenting training models on the preprocessed the TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048) into a single, clean dataset.
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import os                                  # os handles directory/workspace changes
import comet_ml                            # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                               # PyTorch to create and apply deep learning models
import sys
# -

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'data/TCGA-Pancancer/cleaned/'
# Add path to the project scripts
sys.path.append('code/tcga-cancer-classification/scripts/')

import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods
import Models                              # Machine learning models
from Tabular_Dataset import Tabular_Dataset

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# ## Loading the data

tcga_df = pd.read_csv(f'{data_path}normalized/tcga.csv')
tcga_df.head()

tcga_df.sample_id.value_counts()

tcga_df.dtypes

# Remove the original string ID column and use the numeric one instead:

tcga_df = tcga_df.drop(columns=['sample_id'], axis=1)
tcga_df = tcga_df.rename(columns={'Unnamed: 0': 'sample_id'})
tcga_df.head()

# Convert the label to a numeric format:

tcga_df.tumor_type_label.value_counts()

tcga_df['tumor_type_label'], label_dict = du.embedding.enum_categorical_feature(tcga_df, 'tumor_type_label', nan_value=None, 
                                                                                forbidden_digit=None, clean_name=False)
tcga_df.tumor_type_label.value_counts()

label_dict

tcga_df.dtypes

# Convert to a PyTorch tensor:

tcga_tsr = torch.from_numpy(tcga_df.to_numpy())
tcga_tsr

# Create a dataset:

dataset = Tabular_Dataset(tcga_tsr, tcga_df)

len(dataset)

dataset.label_column

dataset.y

# Get the train, validation and test sets data loaders, which will allow loading batches:

batch_size = 32

train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                          batch_size=batch_size, get_indeces=False)

# ## Training models

# Training hyperparameters:

n_epochs = 20                                   # Number of epochs
lr = 0.001                                      # Learning rate

# ### MLP with embedding layer

# Model hyperparameters:

n_ids = tcga_df.sample_id.nunique()           # Total number of sequences
n_inputs = len(tcga_df.columns)               # Number of input features
n_hidden = [100]    # Number of hidden units
n_outputs = tcga_df.tumor_type_label.nunique() # Number of outputs
n_layers = 2                                  # Number of MLP layers
p_dropout = 0.2                               # Probability of dropout
use_batch_norm = False                        # Indicates if batch normalization is applied
embedding_dim = [3, 3]                        # Embedding dimensions for each categorical feature

# Subtracting 1 because of the removed label column, which was before these columns
embed_features = [du.search_explore.find_col_idx(tcga_df, 'race')-1,
                  du.search_explore.find_col_idx(tcga_df, 'ajcc_pathologic_tumor_stage')-1]
embed_features

# **Coments on the number of embeddings:**
# * It's important to consider the maximum encoding number instead of the amount of unique encodings, as there might be numbers that are skipped in between.
# * We have to add 1 as, because we're considering 0 as the representation of NaN/other/separator, the encodings start at the number 1.

num_embeddings = [tcga_df.race.max()+1,
                  tcga_df.ajcc_pathologic_tumor_stage.max()+1]
num_embeddings

tcga_tsr[:, embed_features]

# Initializing the model:

model = Models.MLP(n_inputs-2, n_hidden, n_outputs, n_layers, p_dropout, use_batch_norm,
                   embed_features, num_embeddings, embedding_dim)
model

# Training and testing:

# + {"pixiedust": {"displayParams": {}}}
model, val_loss_min = du.machine_learning.train(model, train_dataloader, val_dataloader, cols_to_remove=0,
                                                model_type='mlp', batch_size=batch_size, n_epochs=n_epochs, 
                                                lr=lr, model_path='code/tcga-cancer-classification/models/',
                                                ModelClass=Models.MLP, do_test=True, log_comet_ml=True,
                                                comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                                comet_ml_project_name='tcga-tumor-classification',
                                                comet_ml_workspace='andrecnf',
                                                comet_ml_save_model=True, features_list=list(tcga_df.columns),
                                                get_val_loss_min=True)
print(f'Minimium validation loss: {val_loss_min}')
# -

