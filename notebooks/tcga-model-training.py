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
import xgboost as xgb                      # Gradient boosting trees models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import joblib                              # Save scikit-learn models in disk
from datetime import datetime              # datetime to use proper date and time formats
import sys
# -

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'data/TCGA-Pancancer/cleaned/'
# Path to the trained models
models_path = 'code/tcga-cancer-classification/models/'
# Path to the hyperparameter optimization configuration files
hyper_opt_config_path = 'code/tcga-cancer-classification/hyperparameter_optimization/'
# Add path to the project scripts
sys.path.append('code/tcga-cancer-classification/scripts/')

import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods
import Models                              # Machine learning models

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# ## Loading the data

tcga_df = pd.read_csv(f'{data_path}normalized/tcga.csv')
tcga_df.head()

tcga_df.info()

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

dataset = du.datasets.Tabular_Dataset(tcga_tsr, tcga_df)

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

# #### Normal training

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
                                                lr=lr, model_path=f'{models_path}mlp/',
                                                ModelClass=Models.MLP, do_test=True, log_comet_ml=True,
                                                comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                                comet_ml_project_name='tcga-tumor-classification',
                                                comet_ml_workspace='andrecnf',
                                                comet_ml_save_model=True, features_list=list(tcga_df.columns),
                                                get_val_loss_min=True)
print(f'Minimium validation loss: {val_loss_min}')
# -
# #### Hyperparameter optimization

# ls code/tcga-cancer-classification/hyperparameter_optimization/

# + {"pixiedust": {"displayParams": {}}}
# # %%pixie_debugger
du.machine_learning.optimize_hyperparameters(Models.MLP, du.datasets.Tabular_Dataset, tcga_df,
                                             config_name='tcga_hyperparameter_optimization_config.yaml', 
                                             comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                             comet_ml_project_name='tcga-tumor-classification',
                                             comet_ml_workspace='andrecnf',
                                             n_inputs=len(tcga_df.columns)-2,
                                             id_column=0, label_column=du.search_explore.find_col_idx(tcga_df, feature='tumor_type_label'),
                                             n_outputs=1, model_type='mlp', models_path='models/',
                                             ModelClass=None, array_param=None,
                                             config_path=hyper_opt_config_path, var_seq=False, clip_value=0.5, 
                                             batch_size=32, n_epochs=20, lr=0.001, 
                                             test_train_ratio=0.2, validation_ratio=0.1,
                                             comet_ml_save_model=True)
# -

# ### XGBoost

# #### Adapting the data to XGBoost and Scikit-Learn

# Make a copy of the dataframe:

sckt_tcga_df = tcga_df.copy()
sckt_tcga_df

# Convert categorical columns to string type:

sckt_tcga_df.race = sckt_tcga_df.race.astype(str)
sckt_tcga_df.ajcc_pathologic_tumor_stage = sckt_tcga_df.ajcc_pathologic_tumor_stage.astype(str)

# One hot encode categorical features:

# + {"pixiedust": {"displayParams": {}}}
sckt_tcga_df, new_cols= du.data_processing.one_hot_encoding_dataframe(sckt_tcga_df, columns=['race', 'ajcc_pathologic_tumor_stage'], 
                                                                      clean_name=False, clean_missing_values=False,
                                                                      has_nan=False, join_rows=False,
                                                                      get_new_column_names=True, inplace=True)
new_cols
# -

sckt_tcga_df.head()

# Remove the ID column:

sckt_tcga_df = sckt_tcga_df.drop(columns='sample_id')
sckt_tcga_df.head()

# Convert to a PyTorch tensor:

sckt_tcga_tsr = torch.from_numpy(sckt_tcga_df.to_numpy())
sckt_tcga_tsr

# Create a dataset:

dataset = du.datasets.Tabular_Dataset(sckt_tcga_tsr, sckt_tcga_df)

len(dataset)

dataset.label_column

dataset.y

# Get the train, validation and test sets data loaders, which will allow loading batches:

train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                          batch_size=len(dataset), get_indeces=False)

# Get the full tensors with all the data from each set:

train_features, train_labels = next(iter(train_dataloader))
val_features, val_labels = next(iter(val_dataloader))
test_features, test_labels = next(iter(test_dataloader))

val_features

len(train_features)

# #### Normal training

# Model hyperparameters:

n_class = tcga_df.tumor_type_label.nunique()    # Number of classes
lr = 0.001                                      # Learning rate
objective = 'multi:softmax'                     # Objective function to minimize (in this case, softmax)
eval_metric = 'mlogloss'                        # Metric to analyze (in this case, multioutput negative log likelihood loss)

# Initializing the model:

xgb_model = xgb.XGBClassifier(objective=objective, eval_metric='mlogloss', learning_rate=lr,
                              num_class=n_class, random_state=du.random_seed, seed=du.random_seed)
xgb_model

# Training with early stopping (stops training if the evaluation metric doesn't improve on 5 consequetive iterations):

xgb_model.fit(train_features, train_labels, early_stopping_rounds=5, eval_set=[(val_features, val_labels)])

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}xgb/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(xgb_model, model_filename)

xgb_model = joblib.load(f'{models_path}xgb/checkpoint_16_12_2019_11_39.model')
xgb_model

# Train until the best iteration:

xgb_model = xgb.XGBClassifier(objective=objective, eval_metric='mlogloss', learning_rate=lr,
                              num_class=n_class, random_state=du.random_seed, seed=du.random_seed)
xgb_model

xgb_model.fit(train_features, train_labels, early_stopping_rounds=5, num_boost_round=xgb_model.best_iteration)

# Evaluate on the test set:

pred = xgb_model.predict(test_features)

acc = accuracy_score(test_labels, pred)
acc

f1 = f1_score(test_labels, pred)
f1

pred_proba = xgb_model.predict_proba(test_features)

loss = log_loss(test_labels, pred_proba)
loss

auc = roc_auc_score(test_labels, pred_proba)
auc

# #### Hyperparameter optimization



# ### Logistic Regression

# #### Normal training

# Model hyperparameters:

multi_class = 'multinomial'
solver = 'lbfgs'
penalty = 'l2'
C = 1
max_iter = 1000

# Initializing the model:

logreg_model = LogisticRegression(multi_class=multi_class, solver=solver, penalty=penalty, C=C, max_iter=max_iter, random_state=du.random_seed)
logreg_model

# Training and testing:

logreg_model.fit(train_features, train_labels)

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}logreg/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(logreg_model, model_filename)

logreg_model = joblib.load(f'{models_path}logreg/checkpoint_16_12_2019_02_27.model')
logreg_model

# Evaluate on the test set:

acc = logreg_model.score(test_features, test_labels)
acc

pred = logreg_model.predict(test_features)

f1 = f1_score(test_labels, pred, average='weighted')
f1

pred_proba = logreg_model.predict_proba(test_features)

loss = log_loss(test_labels, pred_proba)
loss

auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# #### Hyperparameter optimization



# ### SVM

# #### Normal training

# Model hyperparameters:

decision_function_shape = 'ovo'
C = 1
kernel = 'rbf'
max_iter = 100

# Initializing the model:

svm_model = SVC(kernel=kernel, decision_function_shape=decision_function_shape, C=C, 
                max_iter=max_iter, probability=True, random_state=du.random_seed)
svm_model

# Training and testing:

svm_model.fit(train_features, train_labels)

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}svm/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(svm_model, model_filename)

svm_model = joblib.load(f'{models_path}svm/checkpoint_16_12_2019_05_51.model')
svm_model

# Evaluate on the test set:

acc = logreg_model.score(test_features, test_labels)
acc

pred = logreg_model.predict(test_features)

f1 = f1_score(test_labels, pred, average='weighted')
f1

pred_proba = logreg_model.predict_proba(test_features)

loss = log_loss(test_labels, pred_proba)
loss

auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# #### Hyperparameter optimization


