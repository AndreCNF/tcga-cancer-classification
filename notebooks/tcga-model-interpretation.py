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

# # TCGA Model interpretation
# ---
#
# Interpreting the trained machine learning models, retrieving and analysing feature importance. The models were trained on the preprocessed the TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048) into a single, clean dataset.
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import os                                  # os handles directory/workspace changes
import torch                               # PyTorch to create and apply deep learning models
import xgboost as xgb                      # Gradient boosting trees models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import joblib                              # Save scikit-learn models in disk
from datetime import datetime              # datetime to use proper date and time formats
import sys
import numpy as np                         # NumPy for simple mathematical operations
import shap                                # Interpretability package with intuitive plots
import pickle                              # Module to save python objects on disk
# -

# Initialize the javascript visualization library to allow for shap plots:

shap.initjs()

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'data/TCGA-Pancancer/cleaned/'
# Path to the trained models
models_path = 'code/tcga-cancer-classification/models/'
# Path where the model interpreter will be saved
interpreter_path = 'code/tcga-cancer-classification/interpreters/'
# Add path to the project scripts
sys.path.append('code/tcga-cancer-classification/scripts/')

import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods
from model_interpreter.model_interpreter import ModelInterpreter  # Model interpretability class
import Models                              # Machine learning models

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

dataset = du.datasets.Tabular_Dataset(tcga_tsr, tcga_df)

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

# ## Denormalizing the data
#
# [TODO]

tcga_df.columns

# ## Adapting the data to XGBoost and Scikit-Learn

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

sckt_dataset = du.datasets.Tabular_Dataset(sckt_tcga_tsr, sckt_tcga_df)

len(sckt_dataset)

sckt_dataset.label_column

sckt_dataset.y

# Get the train, validation and test sets data loaders, which will allow loading batches:

sckt_train_dataloader, sckt_val_dataloader, sckt_test_dataloader = du.machine_learning.create_train_sets(sckt_dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                                         batch_size=len(sckt_dataset), get_indeces=False)

# Get the full tensors with all the data from each set:

sckt_train_features, sckt_train_labels = next(iter(sckt_train_dataloader))
sckt_val_features, sckt_val_labels = next(iter(sckt_val_dataloader))
sckt_test_features, sckt_test_labels = next(iter(sckt_test_dataloader))

sckt_val_features

len(sckt_train_features)

# ## Interpreting MLP

# ### Loading the model

# + {"pixiedust": {"displayParams": {}}}
model = du.deep_learning.load_checkpoint(filepath=f'{models_path}checkpoint_08_12_2019_05_24.pth', ModelClass=Models.MLP)
model
# -

# Check performance metrics:

output, metrics = du.deep_learning.model_inference(model, dataloader=test_dataloader, metrics=['loss', 'accuracy', 'AUC', 'AUC_weighted'],
                                                   model_type='mlp', cols_to_remove=0)
metrics

# ### Interpreting the model

# Names of the features:

feat_names = list(tcga_df.columns)
feat_names.remove('sample_id')
feat_names.remove('tumor_type_label')

feat_names

# Create a model interpreter object:

interpreter = ModelInterpreter(model, data=dataset.X, labels=dataset.y, model_type='mlp',
                               id_column=0, inst_column=None,
                               fast_calc=True, SHAP_bkgnd_samples=500,
                               random_seed=42, feat_names=feat_names)
interpreter

# Calculate feature importance:

# + {"pixiedust": {"displayParams": {}}}
interpreter.interpret_model(bkgnd_data=train_features, test_data=test_features[:1000], test_labels=test_labels,
                            new_data=False, model_type='mlp', instance_importance=False, 
                            feature_importance='shap', fast_calc=True, see_progress=True, save_data=False, 
                            debug_loss=False)
interpreter.feat_scores
# -

# Path where the model interpreter will be saved
interpreter_path = 'code/tcga-cancer-classification/interpreters/'

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}checkpoint_{current_datetime}.pickle'
# Save model interpreter object, with the instance and feature importance scores, in a pickle file
joblib.dump(interpreter, interpreter_filename)

interpreter = joblib.load(interpreter_filename)
interpreter

# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

len(interpreter.feat_scores)

[scores.sum() for scores in interpreter.feat_scores]

# It appears that only the predicted class (the output with highest score) has the feature importance scores summing to one.

np.argmax([scores.sum() for scores in interpreter.feat_scores])

pred = model(test_features[0, 1:].unsqueeze(0)).topk(1).indices.item()
pred

int(test_labels[0])

# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# #### Summary plot

np.array(interpreter.feat_scores).shape

test_features[0, 1:].unsqueeze(0).numpy().shape

# Summary plot with all 33 tumor types (i.e. all output classes):

test_features[:1000, 1:].unsqueeze(0).numpy().shape

shap.summary_plot(interpreter.feat_scores, 
                  features=test_features[:1000, 1:].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# Summary plot for just one specific tumor type (i.e. just one output class):

output_class = 21

shap.summary_plot(np.array(interpreter.feat_scores)[output_class], 
                  features=test_features[:1000, 1:].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# #### Sample force plot

sample = 0

# +
# [TODO] Not being able to do force plots on multiple outputs
# shap.force_plot(interpreter.explainer.expected_value, 
#                 interpreter.feat_scores, 
#                 features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                 feature_names=feat_names)
# -

shap.force_plot(interpreter.explainer.expected_value[pred], 
                interpreter.feat_scores[pred], 
                features=test_features[sample, 1:].unsqueeze(0).numpy(), 
                feature_names=feat_names)

# #### Sample decision plot

sample = 0

# +
# [TODO] Not being able to do decision plots on multiple outputs
# shap.multioutput_decision_plot(interpreter.explainer.expected_value, 
#                                interpreter.feat_scores, row_index=0,
#                                features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                                feature_names=feat_names)
# -

shap.decision_plot(interpreter.explainer.expected_value[pred], 
                   interpreter.feat_scores[pred], 
                   features=test_features[sample, 1:].unsqueeze(0).numpy(), 
                   feature_names=feat_names)

test_features[:, 1:].shape

test_features[0].shape

test_features[0].unsqueeze(0).shape

# ## Interpreting XGBoost

# ### Loading the model

# + {"pixiedust": {"displayParams": {}}}
xgb_model = joblib.load(f'{models_path}xgb/checkpoint_16_12_2019_11_39.model')
xgb_model
# -

# Check performance metrics:

pred = xgb_model.predict(sckt_test_features)

acc = accuracy_score(sckt_test_labels, pred)
acc

f1 = f1_score(sckt_test_labels, pred, average='weighted')
f1

pred_proba = xgb_model.predict_proba(sckt_test_features)

loss = log_loss(sckt_test_labels, pred_proba)
loss

auc = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# ### Interpreting the model

# Names of the features:

feat_names = list(sckt_tcga_df.columns)
feat_names.remove('tumor_type_label')

# +
# feat_names
# -

# Create a SHAP explainer object:

# +
# shap.TreeExplainer?

# +
# # shap.KernelExplainer?
# -

explainer = shap.TreeExplainer(logreg_model, sckt_train_features.numpy(), feature_dependence='independent')
# explainer = shap.LinearExplainer(logreg_model, np.zeros((1, len(feat_names))), 
#                                  feature_dependence="independent")
# explainer = shap.KernelExplainer(logreg_model.predict_proba, np.zeros((1, len(feat_names))), link='logit')
# explainer = shap.KernelExplainer(logreg_model.predict_proba, np.zeros((1, len(feat_names))), link='logit', max_bkgnd_samples=100)
# explainer = shap.KernelExplainer(logreg_model.predict_proba, np.zeros((1, len(feat_names))))
# explainer = shap.KernelExplainer(xgb_model.predict_proba, np.zeros((1, len(feat_names))), max_bkgnd_samples=100)
# explainer = shap.KernelExplainer(logreg_model.predict_proba, sckt_train_features.numpy(), link='logit')

# Calculate feature importance:

# + {"pixiedust": {"displayParams": {}}}
# shap_values = explainer.shap_values(sckt_test_features.numpy(), l1_reg='num_features(20)', nsamples=500)
shap_values = explainer.shap_values(sckt_test_features[:1000].numpy(), l1_reg='num_features(10)', nsamples=1000)
shap_values
# -

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}xgb/checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(shap_values, interpreter_filename)

shap_values = joblib.load(interpreter_filename)
shap_values

# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

len(shap_values)

[scores.sum() for scores in shap_values]

# It appears that only the predicted class (the output with highest score) has the feature importance scores summing to one.

np.argmax([scores.sum() for scores in shap_values])

pred = model(sckt_test_features[0].unsqueeze(0)).topk(1).indices.item()
pred

int(test_labels[0])

# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# #### Summary plot

np.array(shap_values).shape

shap_values.shape

sckt_test_features[0].unsqueeze(0).numpy().shape

# Summary plot with all 33 tumor types (i.e. all output classes):

sckt_test_features[:1000].unsqueeze(0).numpy().shape

shap.summary_plot(shap_values, 
                  features=sckt_test_features[:1000].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# Summary plot for just one specific tumor type (i.e. just one output class):

output_class = 21

shap.summary_plot(shap_values[output_class], 
                  features=sckt_test_features[:1000].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# #### Sample force plot

sample = 0

# +
# [TODO] Not being able to do force plots on multiple outputs
# shap.force_plot(interpreter.explainer.expected_value, 
#                 interpreter.feat_scores, 
#                 features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                 feature_names=feat_names)
# -

shap.force_plot(shap_values.expected_value[pred], 
                shap_values[pred], 
                features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                feature_names=feat_names)

# #### Sample decision plot

sample = 0

# +
# [TODO] Not being able to do decision plots on multiple outputs
# shap.multioutput_decision_plot(interpreter.explainer.expected_value, 
#                                interpreter.feat_scores, row_index=0,
#                                features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                                feature_names=feat_names)
# -

shap.decision_plot(shap_values.expected_value[pred], 
                   shap_values[pred], 
                   features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                   feature_names=feat_names)

sckt_test_features[:].shape

sckt_test_features[0].shape

sckt_test_features[0].unsqueeze(0).shape

# ## Interpreting Logistic Regression

# ### Loading the model

# + {"pixiedust": {"displayParams": {}}}
logreg_model = joblib.load(f'{models_path}logreg/checkpoint_16_12_2019_02_27.model')
logreg_model
# -

# Check performance metrics:

acc = logreg_model.score(sckt_test_features, sckt_test_labels)
acc

pred = logreg_model.predict(sckt_test_features)

f1 = f1_score(sckt_test_labels, pred, average='weighted')
f1

pred_proba = logreg_model.predict_proba(sckt_test_features)

loss = log_loss(sckt_test_labels, pred_proba)
loss

auc = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# ### Interpreting the model

# Names of the features:

feat_names = list(sckt_tcga_df.columns)
feat_names.remove('tumor_type_label')

# +
# feat_names
# -

# Create a SHAP explainer object:

# +
# # shap.LinearExplainer?

# +
# shap.KernelExplainer?
# -

# explainer = shap.LinearExplainer(logreg_model, sckt_train_features.numpy(), feature_dependence='independent')
# explainer = shap.LinearExplainer(logreg_model, np.zeros((1, len(feat_names))), 
#                                  feature_dependence="independent")
# explainer = shap.KernelExplainer(logreg_model.predict_proba, np.zeros((1, len(feat_names))), link='logit')
# explainer = shap.KernelExplainer(logreg_model.predict_proba, np.zeros((1, len(feat_names))), link='logit', max_bkgnd_samples=100)
# explainer = shap.KernelExplainer(logreg_model.predict_proba, np.zeros((1, len(feat_names))))
explainer = shap.KernelExplainer(logreg_model.predict_proba, np.zeros((1, len(feat_names))), max_bkgnd_samples=100)
# explainer = shap.KernelExplainer(logreg_model.predict_proba, sckt_train_features.numpy(), link='logit')

# Calculate feature importance:

# + {"pixiedust": {"displayParams": {}}}
# shap_values = explainer.shap_values(sckt_test_features.numpy(), l1_reg='num_features(20)', nsamples=500)
shap_values = explainer.shap_values(sckt_test_features[:1000].numpy(), l1_reg='num_features(10)', nsamples=1000)
shap_values
# -

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}logreg/checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(shap_values, interpreter_filename)

shap_values = joblib.load(f'{interpreter_path}checkpoint_15_12_2019_02_06.pickle')
shap_values

print('Hello world!')

# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

len(shap_values)

[scores.sum() for scores in shap_values]

# It appears that only the predicted class (the output with highest score) has the feature importance scores summing to one.

np.argmax([scores.sum() for scores in shap_values])

pred = logreg_model(sckt_test_features[0].unsqueeze(0)).topk(1).indices.item()
pred

int(test_labels[0])

# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# #### Summary plot

np.array(shap_values).shape

shap_values.shape

sckt_test_features[0].unsqueeze(0).numpy().shape

# Summary plot with all 33 tumor types (i.e. all output classes):

sckt_test_features[:1000].unsqueeze(0).numpy().shape

shap.summary_plot(shap_values, 
                  features=sckt_test_features[:1000].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# Summary plot for just one specific tumor type (i.e. just one output class):

output_class = 21

shap.summary_plot(shap_values[output_class], 
                  features=sckt_test_features[:1000].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# #### Sample force plot

sample = 0

# +
# [TODO] Not being able to do force plots on multiple outputs
# shap.force_plot(interpreter.explainer.expected_value, 
#                 interpreter.feat_scores, 
#                 features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                 feature_names=feat_names)
# -

shap.force_plot(shap_values.expected_value[pred], 
                shap_values[pred], 
                features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                feature_names=feat_names)

# #### Sample decision plot

sample = 0

# +
# [TODO] Not being able to do decision plots on multiple outputs
# shap.multioutput_decision_plot(interpreter.explainer.expected_value, 
#                                interpreter.feat_scores, row_index=0,
#                                features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                                feature_names=feat_names)
# -

shap.decision_plot(shap_values.expected_value[pred], 
                   shap_values[pred], 
                   features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                   feature_names=feat_names)

sckt_test_features[:].shape

sckt_test_features[0].shape

sckt_test_features[0].unsqueeze(0).shape

# ## Interpreting SVM

# ### Loading the model

# + {"pixiedust": {"displayParams": {}}}
svm_model = joblib.load(f'{models_path}svm/checkpoint_16_12_2019_05_51.model')
svm_model
# -

# Check performance metrics:

acc = svm_model.score(sckt_test_features, sckt_test_labels)
acc

pred = svm_model.predict(sckt_test_features)

f1 = f1_score(sckt_test_labels, pred, average='weighted')
f1

pred_proba = svm_model.predict_proba(sckt_test_features)

loss = log_loss(sckt_test_labels, pred_proba)
loss

auc = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# ### Interpreting the model

# Names of the features:

feat_names = list(sckt_tcga_df.columns)
feat_names.remove('tumor_type_label')

# +
# feat_names
# -

# Create a SHAP explainer object:

# +
# # shap.LinearExplainer?

# +
# shap.KernelExplainer?
# -

explainer = shap.KernelExplainer(svm_model.predict_proba, np.zeros((1, len(feat_names))), max_bkgnd_samples=100)

# Calculate feature importance:

# + {"pixiedust": {"displayParams": {}}}
shap_values = explainer.shap_values(sckt_test_features[:1000].numpy(), l1_reg='num_features(10)', nsamples=1000)
shap_values
# -

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}svm/checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(shap_values, interpreter_filename)

shap_values = joblib.load(interpreter_filename)
shap_values

# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

len(shap_values)

[scores.sum() for scores in shap_values]

# It appears that only the predicted class (the output with highest score) has the feature importance scores summing to one.

np.argmax([scores.sum() for scores in shap_values])

pred = svm_model(sckt_test_features[0].unsqueeze(0)).topk(1).indices.item()
pred

int(test_labels[0])

# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# #### Summary plot

np.array(shap_values).shape

shap_values.shape

sckt_test_features[0].unsqueeze(0).numpy().shape

# Summary plot with all 33 tumor types (i.e. all output classes):

sckt_test_features[:1000].unsqueeze(0).numpy().shape

shap.summary_plot(shap_values, 
                  features=sckt_test_features[:1000].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# Summary plot for just one specific tumor type (i.e. just one output class):

output_class = 21

shap.summary_plot(shap_values[output_class], 
                  features=sckt_test_features[:1000].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# #### Sample force plot

sample = 0

# +
# [TODO] Not being able to do force plots on multiple outputs
# shap.force_plot(interpreter.explainer.expected_value, 
#                 interpreter.feat_scores, 
#                 features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                 feature_names=feat_names)
# -

shap.force_plot(shap_values.expected_value[pred], 
                shap_values[pred], 
                features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                feature_names=feat_names)

# #### Sample decision plot

sample = 0

# +
# [TODO] Not being able to do decision plots on multiple outputs
# shap.multioutput_decision_plot(interpreter.explainer.expected_value, 
#                                interpreter.feat_scores, row_index=0,
#                                features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                                feature_names=feat_names)
# -

shap.decision_plot(shap_values.expected_value[pred], 
                   shap_values[pred], 
                   features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                   feature_names=feat_names)

sckt_test_features[:].shape

sckt_test_features[0].shape

sckt_test_features[0].unsqueeze(0).shape


