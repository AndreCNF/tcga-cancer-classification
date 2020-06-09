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
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] {"Collapsed": "false"}
# # TCGA Data Joining
# ---
#
# Joining the preprocessed the TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048) into a single, clean dataset.
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false"}
import os                                  # os handles directory/workspace changes
import torch                               # PyTorch to create and apply deep learning models

# + {"Collapsed": "false"}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false"}
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'data/TCGA-Pancancer/cleaned/'

# + {"Collapsed": "false"}
import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods

# + [markdown] {"Collapsed": "false"}
# Allow pandas to show more columns:

# + {"Collapsed": "false"}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility:

# + {"Collapsed": "false"}
du.set_random_seed(42)

# + [markdown] {"toc-hr-collapsed": false, "Collapsed": "false"}
# ## Joining the normalized data

# + [markdown] {"Collapsed": "false"}
# ### Loading the data

# + {"Collapsed": "false"}
rppa_df = pd.read_csv(f'{data_path}normalized/rppa.csv')
rppa_df.head()

# + {"Collapsed": "false"}
rna_df = pd.read_csv(f'{data_path}normalized/rna.csv')
rna_df.head()

# + {"Collapsed": "false"}
dna_mthltn_df = pd.read_csv(f'{data_path}normalized/dna_methylation.csv')
dna_mthltn_df.head()

# + {"Collapsed": "false"}
mirna_df = pd.read_csv(f'{data_path}normalized/mirna.csv')
mirna_df.head()

# + {"Collapsed": "false"}
copy_num_df = pd.read_csv(f'{data_path}normalized/copy_number_ratio.csv')
copy_num_df.head()

# + {"Collapsed": "false"}
pur_plo_df = pd.read_csv(f'{data_path}normalized/purity_ploidy.csv')
pur_plo_df.head()

# + {"Collapsed": "false"}
cdr_df = pd.read_csv(f'{data_path}normalized/clinical_outcome.csv')
cdr_df.head()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Joining dataframes

# + [markdown] {"Collapsed": "false"}
# #### Checking the length of the ID in the dataframes

# + {"Collapsed": "false"}
rppa_df.sample_id.str.len().describe()

# + {"Collapsed": "false"}
rppa_df[rppa_df.sample_id.str.len() == 19]

# + {"Collapsed": "false"}
rna_df.sample_id.str.len().describe()

# + {"Collapsed": "false"}
dna_mthltn_df.sample_id.str.len().describe()

# + {"Collapsed": "false"}
mirna_df.sample_id.str.len().describe()

# + {"Collapsed": "false"}
pur_plo_df.sample_id.str.len().describe()

# + {"Collapsed": "false"}
copy_num_df.sample_id.str.len().describe()

# + {"Collapsed": "false"}
cdr_df.sample_id.str.len().describe()

# + [markdown] {"Collapsed": "false"}
# #### Joining RPPA with RNA data

# + {"Collapsed": "false"}
tcga_df = rppa_df
tcga_df['sample_portion_id'] = tcga_df['sample_id'].str.slice(stop=19)
tcga_df[['sample_id', 'sample_portion_id']].head()

# + {"Collapsed": "false"}
rna_df['sample_portion_id'] = rna_df['sample_id'].str.slice(stop=19)
rna_df[['sample_id', 'sample_portion_id']].head()

# + {"Collapsed": "false"}
# Only 13 matches; ignore RPPA data, at least for now
tcga_df = tcga_df.merge(rna_df, how='inner', on='sample_portion_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining RNA with DNA Methylation data

# + {"Collapsed": "false"}
# 0 matches
tcga_df = rna_df.merge(dna_mthltn_df, how='inner', on='sample_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining RNA with miRNA data

# + {"Collapsed": "false"}
# Only 705 matches
tcga_df = rna_df.merge(mirna_df, how='inner', on='sample_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining RNA with purity/ploidy data

# + {"Collapsed": "false"}
# 0 matches
tcga_df = rna_df.merge(pur_plo_df, how='inner', on='sample_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining DNA Methylation with miRNA data

# + {"Collapsed": "false"}
# 0 matches
tcga_df = dna_mthltn_df.merge(mirna_df, how='inner', on='sample_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining DNA Methylation with purity/ploidy data

# + {"Collapsed": "false"}
# 0 matches
tcga_df = dna_mthltn_df.merge(pur_plo_df, how='inner', on='sample_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining miRNA with purity/ploidy data

# + {"Collapsed": "false"}
# 0 matches
tcga_df = mirna_df.merge(pur_plo_df, how='inner', on='sample_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining RNA with copy number ratio data

# + {"Collapsed": "false"}
tcga_df = rna_df
tcga_df['sample_cpy_id'] = tcga_df['sample_id'].str.slice(stop=15)
tcga_df[['sample_id', 'sample_cpy_id']].head()

# + {"Collapsed": "false"}
copy_num_df['sample_cpy_id'] = copy_num_df['sample_id'].str.slice(stop=15)
copy_num_df[['sample_id', 'sample_cpy_id']].head()

# + {"Collapsed": "false"}
# 9848 matches! Now that's more like it!
tcga_df = tcga_df.merge(copy_num_df, how='inner', on='sample_cpy_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# #### Joining RNA with copy number ratio and with clinical data

# + {"Collapsed": "false"}
tcga_df['participant_id'] = tcga_df['sample_id_x'].str.slice(stop=12)
tcga_df[['sample_id_x', 'participant_id']].head()

# + {"Collapsed": "false"}
cdr_df['participant_id'] = cdr_df['sample_id'].str.slice(stop=12)
cdr_df[['sample_id', 'participant_id']].head()

# + {"Collapsed": "false"}
# 9825 matches! Now that's more like it!
tcga_df = tcga_df.merge(cdr_df, how='inner', on='participant_id')
tcga_df

# + [markdown] {"Collapsed": "false"}
# ### Removing redundant ID columns

# + {"Collapsed": "false"}
id_columns = [col for col in tcga_df if '_id' in col]
id_columns

# + {"Collapsed": "false"}
tcga_df[id_columns].head()

# + {"Collapsed": "false"}
id_columns.remove('participant_id')
tcga_df = tcga_df.drop(columns=id_columns)
tcga_df

# + {"Collapsed": "false"}
id_columns = [col for col in tcga_df if '_id' in col]
id_columns

# + {"Collapsed": "false"}
tcga_df[id_columns].head()

# + [markdown] {"Collapsed": "false"}
# ### Removing repeated patient data
#
# In order to prevent the machine learning models from overfitting to specific patients, we'll randomly select a single sample from patients that have multiple ones, guaranteeing that each patient has only one sample.

# + {"Collapsed": "false"}
tcga_df[tcga_df.participant_id == 'TCGA-SR-A6MX']

# + {"Collapsed": "false"}
tcga_df[tcga_df.participant_id == 'TCGA-SR-A6MX'].sample(n=1, random_state=du.random_seed)

# + {"Collapsed": "false"}
n_samples_per_patient = tcga_df.participant_id.value_counts()
n_samples_per_patient

# + {"Collapsed": "false"}
list(n_samples_per_patient.index)

# + {"Collapsed": "false"}
oversampled_participants = [participant for participant in list(n_samples_per_patient.index)
                            if n_samples_per_patient[participant] > 1]
oversampled_participants

# + {"Collapsed": "false"}
len(oversampled_participants)

# + {"Collapsed": "false"}
# Replace the multiple samples of the same patient with just one, randomly selected, sample
for participant in oversampled_participants:
    # Randomly select one sample from the current patient's data
    new_row = tcga_df[tcga_df.participant_id == participant].sample(n=1, random_state=du.random_seed)
    # Remove all the patient's data
    tcga_df = tcga_df[tcga_df.participant_id != participant]
    # Add just the randomly selected sample
    tcga_df = tcga_df.append(new_row)


# + {"Collapsed": "false"}
tcga_df.participant_id.value_counts()

# + {"Collapsed": "false"}
tcga_df.participant_id.value_counts()[['TCGA-SR-A6MX', 'TCGA-06-0211', 'TCGA-06-0125', 
                                       'TCGA-EM-A3FQ', 'TCGA-E2-A15A']]

# + {"Collapsed": "false"}
tcga_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Performing imputation

# + [markdown] {"Collapsed": "false"}
# Checking for missing values:

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(tcga_df)

# + [markdown] {"Collapsed": "false"}
# Remove columns with too high percentage of missing values (>40%):

# + {"Collapsed": "false"}
tcga_df = du.data_processing.remove_cols_with_many_nans(tcga_df, nan_percent_thrsh=40, inplace=True)
du.search_explore.dataframe_missing_values(tcga_df)

# + [markdown] {"Collapsed": "false"}
# Imputation:

# + {"Collapsed": "false"}
tcga_df = du.data_processing.missing_values_imputation(tcga_df, method='zero',
                                                       id_column='participant_id', inplace=True)
tcga_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Saving the data

# + {"Collapsed": "false"}
tcga_df.to_csv(f'{data_path}normalized/tcga.csv')

# + [markdown] {"Collapsed": "false"}
# ### Experimenting with tensor conversion

# + {"Collapsed": "false"}
tcga_df = pd.read_csv(f'{data_path}normalized/tcga.csv')
tcga_df.head()

# + {"Collapsed": "false"}
tcga_df.participant_id.value_counts()

# + {"Collapsed": "false"}
tcga_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Remove the original string ID column and use the numeric one instead:

# + {"Collapsed": "false"}
tcga_df = tcga_df.drop(columns=['participant_id'], axis=1)
tcga_df = tcga_df.rename(columns={'Unnamed: 0': 'participant_id'})
tcga_df.head()

# + [markdown] {"Collapsed": "false"}
# Convert the label to a numeric format:

# + {"Collapsed": "false"}
tcga_df.tumor_type_label.value_counts()

# + {"Collapsed": "false"}
tcga_df['tumor_type_label'], label_dict = du.embedding.enum_categorical_feature(tcga_df, 'tumor_type_label', 
                                                                                nan_value=0, clean_name=False)
tcga_df.tumor_type_label.value_counts()

# + {"Collapsed": "false"}
label_dict

# + {"Collapsed": "false"}
tcga_df.dtypes

# + {"Collapsed": "false"}
tcga_tsr = torch.from_numpy(tcga_df.to_numpy())
tcga_tsr

# + {"Collapsed": "false"}

