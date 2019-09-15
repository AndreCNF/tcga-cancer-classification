# -*- coding: utf-8 -*-
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

# # TCGA Data Exploration
# ---
#
# Exploring the preprocessed TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048).
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import dask.dataframe as dd                # Dask to handle big data in dataframes
import pandas as pd                        # Pandas to handle the data in dataframes
from dask.distributed import Client        # Dask scheduler
import re                                  # re to do regex searches in string data
import plotly                              # Plotly for interactive and pretty plots
import plotly.graph_objs as go
from datetime import datetime              # datetime to use proper date and time formats
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook             # tqdm allows to track code execution progress
import numbers                             # numbers allows to check if data is numeric
import torch                               # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import utils                               # Contains auxiliary functions
# -

# Set the random seed for reproducibility:

utils.set_random_seed(0)

# Import the remaining custom packages:

import search_explore                      # Methods to search and explore data
import data_processing                     # Data processing and dataframe operations
import embedding                           # Embedding and encoding related methods
# import padding                             # Padding and variable sequence length related methods
# import machine_learning                    # Common and generic machine learning related methods
# import deep_learning                       # Common and generic deep learning related methods

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# +
# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the dataset files
data_path = 'Datasets/Scholarship/TCGA-Pancancer/'
file1_folder = 'fcbb373e-28d4-4818-92f3-601ede3da5e1/'
file2_folder = 'd82e2c44-89eb-43d9-b6d3-712732bf6a53/'
file3_folder = '99b0c493-9e94-4d99-af9f-151e46bab989/'
file4_folder = '7d4c0344-f018-4ab0-949a-09815f483480/'
file5_folder = '55d9bf6f-0712-4315-b588-e6f8e295018e/'
file6_folder = '4f277128-f793-4354-a13d-30cc7fe9f6b5/'
file7_folder = '3586c0da-64d0-4b74-a449-5ff4d9136611/'
file8_folder = '1c8cfe5f-e52d-41ba-94da-f15ea1337efc/'
file9_folder = '1c6174d9-8ffb-466e-b5ee-07b204c15cf8/'
file10_folder = '1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/'
file11_folder = '1a7d7be8-675d-4e60-a105-19d4121bdebf/'
file12_folder = '0fc78496-818b-4896-bd83-52db1f533c5c/'
file13_folder = '0f4f5701-7b61-41ae-bda9-2805d1ca9781/'
file14_folder = '00a32f7a-c85f-4f86-850d-be53973cbc4d/'

# Path to the code files
project_path = 'GitHub/tcga-cancer-classification/'
# -

# Set up local cluster
client = Client()
client

# Upload the custom methods files, so that the Dask cluster has access to relevant auxiliary functions
client.upload_file(f'{project_path}NeuralNetwork.py')
client.upload_file(f'{project_path}search_explore.py')
client.upload_file(f'{project_path}data_processing.py')
client.upload_file(f'{project_path}embedding.py')
# client.upload_file(f'{project_path}padding.py')
# client.upload_file(f'{project_path}machine_learning.py')
# client.upload_file(f'{project_path}deep_learning.py')

# + {"colab_type": "text", "id": "bEqFkmlYCGOz", "cell_type": "markdown"}
# **Important:** Use the following two lines to be able to do plotly plots offline:

# + {"colab": {}, "colab_type": "code", "id": "fZCUmUOzCPeI"}
import plotly.offline as py
plotly.offline.init_notebook_mode(connected=True)
# -

# ## Exploring the preprocessed dataset

# ### RPPA data
#
# Reverse phase protein array (RPPA) is a high-throughput antibody-based technique with the procedures similar to that of Western blots. Proteins are extracted from tumor tissue or cultured cells, denatured by SDS , printed on nitrocellulose-coated slides followed by antibody probe. Our RPPA platform currently allows for the analysis of >1000 samples using at least 130 different antibodies.

# rppa_df = pd.read_csv(f'{data_path}{file1_folder}TCGA-RPPA-pancan-clean.csv')
rppa_df = pd.read_csv(f'{data_path}TCGA-RPPA-pancan-clean.csv')
rppa_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# rppa_df = dd.read_csv(f'{data_path}{file1_folder}TCGA-RPPA-pancan-clean.csv')
# rppa_df.head()
# -

# #### Basic stats

rppa_df.dtypes

rppa_df.nunique()

search_explore.dataframe_missing_values(rppa_df)

# Out of all the 200 columns, only 9 of them have missing values, with 8 having more than 49% (`ARID1A`, `ADAR1`, `ALPHACATENIN`, `TTF1`, `PARP1`, `JAB1`, `CASPASE9`, `CASPASE3`).

rppa_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# ### Tumor representation

rppa_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = rppa_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

# This RPPA data has a **very small sample size**, specially considering how big the whole GDC/TCGA dataset really is. Furthermore, **it's significantly unbalenced**, with the most represented tumor type (BRCA) having 892 samples while the least represented tumor type (UVM) has only 12.

# ### RNA data
#
# Description

rna_df = pd.read_csv(f'{data_path}EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep='\t')
rna_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# rna_df = dd.read_csv(f'{data_path}{file1_folder}EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep='\t')
# rna_df.head()
# -

# #### Basic stats

rna_df.dtypes

rna_df.nunique()

search_explore.dataframe_missing_values(rna_df)

#

rna_df.describe().transpose()

#

# ### Tumor representation

rna_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = rna_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#

# ### DNA Methylation data
#
# Description

dna_mthltn_df = pd.read_csv(f'{data_path}jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv', sep='\t')
dna_mthltn_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# dna_mthltn_df = dd.read_csv(f'{data_path}{file1_folder}jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv', sep='\t')
# dna_mthltn_df.head()
# -

# #### Basic stats

dna_mthltn_df.dtypes

dna_mthltn_df.nunique()

search_explore.dataframe_missing_values(dna_mthltn_df)

#

dna_mthltn_df.describe().transpose()

#

# ### Tumor representation

dna_mthltn_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = dna_mthltn_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#

# ### miRNA data
#
# Description

mirna_df = pd.read_csv(f'{data_path}pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv')
mirna_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# mirna_df = dd.read_csv(f'{data_path}{file1_folder}pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv')
# mirna_df.head()
# -

# #### Basic stats

mirna_df.dtypes

mirna_df.nunique()

search_explore.dataframe_missing_values(mirna_df)

#

mirna_df.describe().transpose()

#

# ### Tumor representation

mirna_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = mirna_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#

# ### ABSOLUTE-annotated seg data
#
# Description

abs_anttd_seg_df = pd.read_csv(f'{data_path}TCGA_mastercalls.abs_segtabs.fixed.txt', sep='\t')
abs_anttd_seg_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# abs_anttd_seg_df = dd.read_csv(f'{data_path}{file1_folder}TCGA_mastercalls.abs_segtabs.fixed.txt', sep='\t')
# abs_anttd_seg_df.head()
# -

# #### Basic stats

abs_anttd_seg_df.dtypes

abs_anttd_seg_df.nunique()

search_explore.dataframe_missing_values(abs_anttd_seg_df)

#

abs_anttd_seg_df.describe().transpose()

#

# ### Tumor representation

abs_anttd_seg_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = abs_anttd_seg_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#

# ### ABSOLUTE purity/ploidy data
#
# Description

abs_anttd_pur_df = pd.read_csv(f'{data_path}TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
abs_anttd_pur_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# abs_anttd_pur_df = dd.read_csv(f'{data_path}{file1_folder}TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
# abs_anttd_pur_df.head()
# -

# #### Basic stats

abs_anttd_pur_df.dtypes

abs_anttd_pur_df.nunique()

search_explore.dataframe_missing_values(abs_anttd_pur_df)

#

abs_anttd_pur_df.describe().transpose()

#

# ### Tumor representation

abs_anttd_pur_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = abs_anttd_pur_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#

# ### Mutations data
#
# Description

mut_df = pd.read_csv(f'{data_path}mc3.v0.2.8.PUBLIC.maf.gz', sep='\t')
mut_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# mut_df = dd.read_csv(f'{data_path}{file1_folder}mc3.v0.2.8.PUBLIC.maf.gz', sep='\t')
# mut_df.head()
# -

# #### Basic stats

mut_df.dtypes

mut_df.nunique()

search_explore.dataframe_missing_values(mut_df)

#

mut_df.describe().transpose()

#

# ### Tumor representation

mut_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = mut_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#

# ### Clinical outcome (TCGA-CDR) data
#
# Description

cdr_df = pd.read_excel(f'{data_path}TCGA-CDR-SupplementalTableS1.xlsx')
cdr_df.head()

# +
# For some reason Dask is failing to read the excel file directly (failed to serialize)
# cdr_df = dd.read_excel(f'{data_path}{file1_folder}TCGA-CDR-SupplementalTableS1.xlsx')
# cdr_df.head()
# -

# #### Basic stats

cdr_df.dtypes

cdr_df.nunique()

search_explore.dataframe_missing_values(cdr_df)

#

cdr_df.describe().transpose()

#

# ### Tumor representation

cdr_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = cdr_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#

# ### Clinical with follow-up data
#
# Description

clnc_fllw_df = pd.read_csv(f'{data_path}clinical_PANCAN_patient_with_followup.tsv', sep='\t')
clnc_fllw_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# clnc_fllw_df = dd.read_csv(f'{data_path}{file1_folder}clinical_PANCAN_patient_with_followup.tsv', sep='\t')
# clnc_fllw_df.head()
# -

# #### Basic stats

clnc_fllw_df.dtypes

clnc_fllw_df.nunique()

search_explore.dataframe_missing_values(clnc_fllw_df)

#

clnc_fllw_df.describe().transpose()

#

# ### Tumor representation

clnc_fllw_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = clnc_fllw_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

#
