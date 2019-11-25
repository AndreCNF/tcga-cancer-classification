# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TCGA Data Joining
# ---
#
# Joining the preprocessed the TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048) into a single, clean dataset.
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import os                                  # os handles directory/workspace changes
# -

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'storage/data/TCGA-Pancancer/cleaned/'

import modin.pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Joining the normalized data
# -

# ### Loading the data

rppa_df = pd.read_csv(f'{data_path}normalized/rppa.csv')
rppa_df.head()

rna_df = pd.read_csv(f'{data_path}normalized/rna.csv')
rna_df.head()

dna_mthltn_df = pd.read_csv(f'{data_path}normalized/dna_methylation.csv')
dna_mthltn_df.head()

mirna_df = pd.read_csv(f'{data_path}normalized/mirna.csv')
mirna_df.head()

# [TODO] Something's wrong with this dataframe too, as the `sample_id` column doesn't uniquely identify rows; fix it
copy_num_df = pd.read_csv(f'{data_path}normalized/copy_number_ratio.csv')
copy_num_df.head()

pur_plo_df = pd.read_csv(f'{data_path}normalized/purity_ploidy.csv')
pur_plo_df.head()

cdr_df = pd.read_csv(f'{data_path}normalized/clinical_outcome.csv')
cdr_df.head()

# ### Joining dataframes



# ### Performing imputation



# ### Experimenting with tensor conversion



# ### Saving the data


