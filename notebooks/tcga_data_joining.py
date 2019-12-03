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

# # TCGA Data Joining
# ---
#
# Joining the preprocessed the TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048) into a single, clean dataset.
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import os                                  # os handles directory/workspace changes
# -

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'data/TCGA-Pancancer/cleaned/'

import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# + [markdown] {"toc-hr-collapsed": true}
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

copy_num_df = pd.read_csv(f'{data_path}normalized/copy_number_ratio.csv')
copy_num_df.head()

pur_plo_df = pd.read_csv(f'{data_path}normalized/purity_ploidy.csv')
pur_plo_df.head()

cdr_df = pd.read_csv(f'{data_path}normalized/clinical_outcome.csv')
cdr_df.head()

# ### Joining dataframes

# #### Checking the length of the ID in the dataframes

rppa_df.sample_id.str.len().describe()

rppa_df[rppa_df.sample_id.str.len() == 19]

rna_df.sample_id.str.len().describe()

dna_mthltn_df.sample_id.str.len().describe()

mirna_df.sample_id.str.len().describe()

pur_plo_df.sample_id.str.len().describe()

copy_num_df.sample_id.str.len().describe()

cdr_df.sample_id.str.len().describe()

# #### Joining RPPA with RNA data

tcga_df = rppa_df
tcga_df['sample_portion_id'] = tcga_df['sample_id'].str.slice(stop=19)
tcga_df[['sample_id', 'sample_portion_id']].head()

rna_df['sample_portion_id'] = rna_df['sample_id'].str.slice(stop=19)
rna_df[['sample_id', 'sample_portion_id']].head()

# Only 13 matches; ignore RPPA data, at least for now
tcga_df = tcga_df.merge(rna_df, how='inner', on='sample_portion_id')
tcga_df

# #### Joining RNA with DNA Methylation data

# 0 matches
tcga_df = rna_df.merge(dna_mthltn_df, how='inner', on='sample_id')
tcga_df

# #### Joining RNA with miRNA data

# Only 705 matches
tcga_df = rna_df.merge(mirna_df, how='inner', on='sample_id')
tcga_df

# #### Joining RNA with purity/ploidy data

# 0 matches
tcga_df = rna_df.merge(pur_plo_df, how='inner', on='sample_id')
tcga_df

# #### Joining DNA Methylation with miRNA data

# 0 matches
tcga_df = dna_mthltn_df.merge(mirna_df, how='inner', on='sample_id')
tcga_df

# #### Joining DNA Methylation with purity/ploidy data

# 0 matches
tcga_df = dna_mthltn_df.merge(pur_plo_df, how='inner', on='sample_id')
tcga_df

# #### Joining miRNA with purity/ploidy data

# 0 matches
tcga_df = mirna_df.merge(pur_plo_df, how='inner', on='sample_id')
tcga_df

# #### Joining RNA with copy number ratio data

tcga_df = rna_df
tcga_df['sample_cpy_id'] = tcga_df['sample_id'].str.slice(stop=15)
tcga_df[['sample_id', 'sample_cpy_id']].head()

copy_num_df['sample_cpy_id'] = copy_num_df['sample_id'].str.slice(stop=15)
copy_num_df[['sample_id', 'sample_cpy_id']].head()

# 9870 matches! Now that's more like it!
tcga_df = tcga_df.merge(copy_num_df, how='inner', on='sample_cpy_id')
tcga_df

# #### Joining RNA with copy number ratio and with clinical data

tcga_df['participant_id'] = tcga_df['sample_id_x'].str.slice(stop=12)
tcga_df[['sample_id_x', 'participant_id']].head()

cdr_df['participant_id'] = cdr_df['sample_id'].str.slice(stop=12)
cdr_df[['sample_id', 'participant_id']].head()

# 9847 matches! Now that's more like it!
tcga_df = tcga_df.merge(cdr_df, how='inner', on='participant_id')
tcga_df

# #### Removing redundant ID columns

id_columns = [col for col in tcga_df if '_id' in col]
id_columns

id_columns.remove('sample_id')
tcga_df = tcga_df.drop(columns=id_columns)
tcga_df

# ### Performing imputation

# Checking for missing values:

du.search_explore.dataframe_missing_values(tcga_df)

# Remove columns with too high percentage of missing values (>40%):

tcga_df = du.data_processing.remove_cols_with_many_nans(tcga_df, nan_percent_thrsh=40, inplace=True)
du.search_explore.dataframe_missing_values(tcga_df)

# Imputation:

tcga_df = du.data_processing.missing_values_imputation(tcga_df, method='interpolation',
                                                       id_column='sample_id', inplace=True)
tcga_df.head()

# ### Saving the data

tcga_df.to_csv(f'{data_path}normalized/tcga.csv')

# ### Experimenting with tensor conversion


