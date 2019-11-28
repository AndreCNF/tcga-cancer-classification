# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: tcga-cancer-classification
#     language: python
#     name: tcga-cancer-classification
# ---

# # Models dummy tests
#
# Testing models from the project defined classes, including the embedding layers and time intervals handling, on dummy datasets.

# ## Importing the necessary packages

import pandas as pd                        # Pandas to load the data initially
import numpy as np                         # Mathematical operations package, allowing also for missing values representation
import torch                               # PyTorch for tensor and deep learning operations
import data_utils as du                    # Data science and machine learning relevant methods
import os                                  # os handles directory/workspace changes

du.random_seed

du.set_random_seed(42)

du.random_seed

du.use_modin

du.set_pandas_library('pandas')

du.use_modin

import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to scripts directory
os.chdir('../../scripts')

from Tabular_Dataset import Tabular_Dataset # Dataset class that helps fetching batches of data
import Models                              # Script with all the machine learning model classes

# Change to parent directory (presumably "eICU-mortality-prediction")
os.chdir('..')

# ## Initializing variables

# Data that we'll be using:

# + {"pixiedust": {"displayParams": {}}}
dmy_data = np.array([[0, 23, 284, 70, 5, np.nan, 0],
                     [91, 23, 284, 70, 5, 'b', 0],
                     [92, 24, 270, 73, 5, 'b', 0],
                     [93, 22, 290, 71, 5, 'a', 0],
                     [93, 22, 290, 71, 5, 'b', 0],
                     [94, 20, 288, 65, 4, 'a', 1],
                     [94, 20, 288, 65, 4, 'b', 1],
                     [95, 21, 297, 64, 4, 'a', 1],
                     [95, 21, 297, 64, 4, 'b', 1],
                     [95, 21, 297, 64, 4, 'c', 1],
                     [10, 25, 300, 76, 5, 'a', 0],
                     [11, 19, 283, 70, 5, 'c', 0],
                     [12, 19, 306, 59, 5, 'a', 1],
                     [12, 19, 306, 59, 5, 'c', 1],
                     [13, 18, 298, 55, 3, 'c', 1],
                     [20, 20, 250, 70, 5, 'c', 0],
                     [21, 20, 254, 68, 4, 'a', 1],
                     [21, 20, 254, 68, 4, 'c', 1],
                     [22, 19, 244, 70, 3, 'a', 1],
                     [30, 27, 264, 78, 4, 'b', 0],
                     [31, 22, 293, 67, 4, 'b', 1]])
# -

dmy_data

dmy_df = pd.DataFrame(dmy_data, columns=['subject_id', 'Var0', 'Var1', 'Var2', 'Var3', 'Var4', 'label'])
dmy_df

dmy_df.dtypes

# Fix the columns dtypes:

dmy_df['subject_id'] = dmy_df['subject_id'].astype(int)
dmy_df['Var0'] = dmy_df['Var0'].astype(int)
dmy_df['Var1'] = dmy_df['Var1'].astype(int)
dmy_df['Var2'] = dmy_df['Var2'].astype(int)
dmy_df['Var3'] = dmy_df['Var3'].astype(int)
dmy_df['Var4'] = dmy_df['Var4'].astype(str)
dmy_df['label'] = dmy_df['label'].astype(int)

dmy_df.dtypes

# +
# List of used features
dmy_cols = list(dmy_df.columns)

# Remove features that aren't used by the model to predict the label
for unused_feature in ['subject_id', 'label']:
    dmy_cols.remove(unused_feature)
# -

dmy_cols

# ## Preparing the dataset

# ### Encoding categories
#
# Converting the categorical feature `Var4` into a numeric format, so that it can be used by the neural networks and by embedding layers.

# Encode each row's categorical value:

# + {"pixiedust": {"displayParams": {}}}
dmy_df['Var4'], enum_dict = du.embedding.enum_categorical_feature(dmy_df, feature='Var4')
dmy_df
# -

enum_dict

# Join the rows and their categories:

dmy_df = du.embedding.join_categorical_enum(dmy_df, cat_feat='Var4', id_columns='subject_id')
dmy_df

# ### Normalizing the features

dmy_df.describe().transpose()

# + {"pixiedust": {"displayParams": {}}}
dmy_norm_df = du.data_processing.normalize_data(dmy_df, id_columns='subject_id',
                                                categ_columns=['Var4', 'label'], see_progress=False)
dmy_norm_df
# -

dmy_norm_df.describe().transpose()

# ### Converting string encodings to numeric

dmy_norm_df = du.embedding.string_encod_to_numeric(dmy_norm_df, cat_feat='Var4', inplace=True)
dmy_norm_df

# ### Dataset object

# +
# # Only run this when testing the MLP, without embeddings
# dmy_norm_df = dmy_norm_df.drop(columns='Var4')
# dmy_norm_df.head()

# + {"pixiedust": {"displayParams": {}}}
dataset = Tabular_Dataset(dmy_norm_df.to_numpy(), dmy_norm_df)
# -

# ### Separating into train and validation sets
#
# Since this notebook is only for experimentation purposes, with a very small dummy dataset, we'll not be using a test set.

# Training parameters:

batch_size = 32                                 # Number of patients in a mini batch
n_epochs = 50                                   # Number of epochs
lr = 0.001                                      # Learning rate

# Separation in train and validation sets:

# Get the train and validation sets data loaders, which will allow loading batches
train_dataloader, val_dataloader, _ = du.machine_learning.create_train_sets(dataset, test_train_ratio=0, 
                                                                            validation_ratio=0.25,
                                                                            batch_size=4, get_indeces=False)

next(iter(train_dataloader))[0]

next(iter(val_dataloader))[0]

# ## Models testing

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ### MLP
# -

# #### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_layers = 2                                  # Number of MLP layers
p_dropout = 0.2                               # Probability of dropout
use_batch_norm = False                        # Indicates if batch normalization is applied

# Instantiating the model:

model = Models.MLP(n_inputs-2, n_hidden, n_outputs, n_layers, p_dropout, use_batch_norm)
model

# #### Training the model

next(model.parameters())

# + {"pixiedust": {"displayParams": {}}}
# Warning: The loss will explode if the feature `Var4` isn't removed or embedded, because of its very high values
model = du.deep_learning.train(model, train_dataloader, val_dataloader, cols_to_remove=0,
                               model_type='mlp', batch_size=batch_size, n_epochs=n_epochs, 
                               lr=lr, model_path='models/', do_test=False, log_comet_ml=False)
# -

next(model.parameters())

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, seq_len_dict, dataloader=val_dataloader, 
                                                   model_type='mlp', metrics=['loss', 'accuracy', 'AUC'],
                                                   output_rounded=False, set_name='test', 
                                                   cols_to_remove=du.search_explore.find_col_idx(dmy_norm_df, 'subject_id'))
output

metrics

# ### MLP with embedding layers

# #### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_layers = 2                                  # Number of LSTM layers
p_dropout = 0.2                               # Probability of dropout
use_batch_norm = False                        # Indicates if batch normalization is applied

# Instantiating the model:

# + {"pixiedust": {"displayParams": {}}}
model = Models.MLP(n_inputs-2, n_hidden, n_outputs, n_layers, p_dropout, use_batch_norm,
                   embed_features=du.search_explore.find_col_idx(dmy_norm_df, 'Var4'), num_embeddings=5,
                   embedding_dim=2)
model
# -

# #### Training the model

next(model.parameters())

# + {"pixiedust": {"displayParams": {}}}
model = du.deep_learning.train(model, train_dataloader, val_dataloader, cols_to_remove=0,
                               model_type='mlp', batch_size=batch_size, n_epochs=n_epochs, 
                               lr=lr, model_path='models/', do_test=False, log_comet_ml=False)
# -

next(model.parameters())

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, seq_len_dict, dataloader=val_dataloader, 
                                                   model_type='mlp', metrics=['loss', 'accuracy', 'AUC'],
                                                   output_rounded=False, set_name='test', 
                                                   cols_to_remove=du.search_explore.find_col_idx(dmy_norm_df, 'subject_id'))
output

metrics

# ### Regularization Learning Network (RLN)



# ### SVM



# ### Decision tree



# ### XGBoost


