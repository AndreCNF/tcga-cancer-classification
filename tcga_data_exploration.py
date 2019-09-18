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

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

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
data_path = 'storage/data/TCGA-Pancancer/'
rppa_folder = 'fcbb373e-28d4-4818-92f3-601ede3da5e1/'
dna_mthltn_folder = 'd82e2c44-89eb-43d9-b6d3-712732bf6a53/'
abs_anttd_pur_folder = '4f277128-f793-4354-a13d-30cc7fe9f6b5/'
rna_folder = '3586c0da-64d0-4b74-a449-5ff4d9136611/'
mut_folder = '1c8cfe5f-e52d-41ba-94da-f15ea1337efc/'
mirna_folder = '1c6174d9-8ffb-466e-b5ee-07b204c15cf8/'
cdr_folder = '1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/'
clnc_fllw_folder = '0fc78496-818b-4896-bd83-52db1f533c5c/'
abs_anttd_seg_folder = '0f4f5701-7b61-41ae-bda9-2805d1ca9781/'

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

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ## Exploring the preprocessed dataset

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### RPPA data
#
# Reverse phase protein array (RPPA) is a high-throughput antibody-based technique with the procedures similar to that of Western blots. Proteins are extracted from tumor tissue or cultured cells, denatured by SDS , printed on nitrocellulose-coated slides followed by antibody probe. Our RPPA platform currently allows for the analysis of >1000 samples using at least 130 different antibodies.
# -

rppa_df = pd.read_csv(f'{data_path}{rppa_folder}TCGA-RPPA-pancan-clean.csv')
# rppa_df = pd.read_csv(f'{data_path}TCGA-RPPA-pancan-clean.csv')
rppa_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# rppa_df = dd.read_csv(f'{data_path}{file1_folder}TCGA-RPPA-pancan-clean.csv')
# rppa_df.head()
# -

# Convert to dask for distributed computing:

rppa_df = dd.from_pandas(rppa_df, npartitions=30)
rppa_df.head()

# #### Basic stats

rppa_df.dtypes

rppa_df.compute().nunique()

search_explore.dataframe_missing_values(rppa_df)

# Out of all the 200 columns, only 9 of them have missing values, with 8 having more than 49% (`ARID1A`, `ADAR1`, `ALPHACATENIN`, `TTF1`, `PARP1`, `JAB1`, `CASPASE9`, `CASPASE3`).

rppa_df = data_processing.standardize_missing_values_df(rppa_df)
rppa_df.head()

search_explore.dataframe_missing_values(rppa_df)

# In this dataset, all missing values were already well represented. We can see this as the missing values percentages didn't change after applying the `standardize_missing_values_df` method.

rppa_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# #### Tumor representation

rppa_df.TumorType.value_counts().to_frame()

data = [go.Histogram(x = rppa_df.TumorType)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

# This RPPA data has a **very small sample size**, specially considering how big the whole GDC/TCGA dataset really is. Furthermore, **it's significantly unbalenced**, with the most represented tumor type (BRCA) having 892 samples while the least represented tumor type (UVM) has only 12.

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### RNA data
#
# Description
# -

rna_df = pd.read_csv(f'{data_path}{rna_folder}EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep='\t')
# rna_df = pd.read_csv(f'{data_path}EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep='\t')
rna_df.head()

# +
# For some reason Dask is failing to read the TSV file directly (failed to serialize)
# rna_df = dd.read_csv(f'{data_path}{rna_folder}EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep='\t', sample=1000000)
# rna_df.head()
# -

# This dataframe is inverted, i.e. the columns should switch with the rows.

rna_df = rna_df.set_index('gene_id').transpose()
rna_df.head()

# Convert to dask for distributed computing:

rna_df = dd.from_pandas(rna_df, npartitions=30)
rna_df.head()

# #### Basic stats

rna_df.dtypes

rna_df.compute().nunique()

search_explore.dataframe_missing_values(rna_df)

# No gene has more than 16% missing values.

# +
# rna_df = data_processing.standardize_missing_values_df(rna_df)
# rna_df.head()

# +
# search_explore.dataframe_missing_values(rna_df)
# -

# The missing values standardization process with take around 30 hours to complete (on Paperspace's C7 machine)! Still, it seems like this table has the right missing values representation, so we don't need to run these last two cells.

rna_df.describe().compute().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### DNA Methylation data
#
# Description
# -

dna_mthltn_df = pd.read_csv(f'{data_path}{dna_mthltn_folder}jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv', sep='\t')
# dna_mthltn_df = pd.read_csv(f'{data_path}jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv', sep='\t')
dna_mthltn_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# dna_mthltn_df = dd.read_csv(f'{data_path}{file1_folder}jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv', sep='\t')
# dna_mthltn_df.head()
# -

# This dataframe is inverted, i.e. the columns should switch with the rows.

dna_mthltn_df = dna_mthltn_df.set_index('Composite Element REF').transpose()
dna_mthltn_df.head()

# Fix the index name:

dna_mthltn_df.index.rename('sample_id', inplace=True)
dna_mthltn_df.head()

# Convert to dask for distributed computing:

dna_mthltn_df = dd.from_pandas(dna_mthltn_df, npartitions=30)
dna_mthltn_df.head()

# #### Basic stats

dna_mthltn_df.dtypes

dna_mthltn_df.compute().nunique()

search_explore.dataframe_missing_values(dna_mthltn_df)

# The DNA composite with the most missing values only has less than 35% missingness.

# +
# dna_mthltn_df = data_processing.standardize_missing_values_df(dna_mthltn_df)
# dna_mthltn_df.head()

# +
# search_explore.dataframe_missing_values(dna_mthltn_df)
# -

# The missing values standardization process with take around 30 hours to complete (on Paperspace's C7 machine)! Still, it seems like this table has the right missing values representation, so we don't need to run these last two cells.

dna_mthltn_df.describe().compute().transpose()

#

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### miRNA data
#
# Description
# -

mirna_df = pd.read_csv(f'{data_path}{mirna_folder}pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv')
# mirna_df = pd.read_csv(f'{data_path}pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv')
mirna_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# mirna_df = dd.read_csv(f'{data_path}{file1_folder}pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv')
# mirna_df.head()
# -

mirna_df.Correction.value_counts()

# Since only 82 genes are "uncorrected" (probably means no preprocessing, such as removing batch effects, was done), we should just remove them. For now, we'll simply drop the `Correction` column.

mirna_df = mirna_df.drop(columns='Correction')
mirna_df.head()

# This dataframe is inverted, i.e. the columns should switch with the rows.

mirna_df = mirna_df.set_index('Genes').transpose()
mirna_df.head()

# Convert to dask for distributed computing:

mirna_df = dd.from_pandas(mirna_df, npartitions=30)
mirna_df.head()

# #### Basic stats

mirna_df.dtypes

mirna_df.compute().nunique()

search_explore.dataframe_missing_values(mirna_df)

# Absolutely no missing values in this dataframe!

mirna_df = data_processing.standardize_missing_values_df(mirna_df)
mirna_df.head()

search_explore.dataframe_missing_values(mirna_df)

#

mirna_df.describe().compute().transpose()

#

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### ABSOLUTE-annotated seg data
#
# This dataframe contains copy-number and copy-ratio related data.
#
# Copy number alterations/aberrations (CNAs) are changes in copy number that have arisen in somatic tissue (for example, just in a tumor), copy number variations (CNVs) originated from changes in copy number in germline cells (and are thus in all cells of the organism).
# -

abs_anttd_seg_df = pd.read_csv(f'{data_path}{abs_anttd_seg_folder}TCGA_mastercalls.abs_segtabs.fixed.txt', sep='\t')
# abs_anttd_seg_df = pd.read_csv(f'{data_path}TCGA_mastercalls.abs_segtabs.fixed.txt', sep='\t')
abs_anttd_seg_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# abs_anttd_seg_df = dd.read_csv(f'{data_path}{file1_folder}TCGA_mastercalls.abs_segtabs.fixed.txt', sep='\t')
# abs_anttd_seg_df.head()
# -

# Convert to dask for distributed computing:

abs_anttd_seg_df = dd.from_pandas(abs_anttd_seg_df, npartitions=30)
abs_anttd_seg_df.head()

# #### Basic stats

abs_anttd_seg_df.dtypes

abs_anttd_seg_df.compute().nunique()

search_explore.dataframe_missing_values(abs_anttd_seg_df)

# Low percentages of missing values, topping at bellow 8%.

abs_anttd_seg_df = data_processing.standardize_missing_values_df(abs_anttd_seg_df)
abs_anttd_seg_df.head()

search_explore.dataframe_missing_values(abs_anttd_seg_df)

# In this dataset, all missing values were already well represented. We can see this as the missing values percentages didn't change after applying the `standardize_missing_values_df` method.

abs_anttd_seg_df.describe().compute().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

abs_anttd_seg_df.solution.value_counts().compute()

# Columns `Start`, `End` and `Num_Probes` don't seem to be relevant in this stationary (not temporal) scenario, without the need for experiment specific information.

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### ABSOLUTE purity/ploidy data
#
# Description
# -

abs_anttd_pur_df = pd.read_csv(f'{data_path}{abs_anttd_pur_folder}TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
# abs_anttd_pur_df = pd.read_csv(f'{data_path}TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
abs_anttd_pur_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# abs_anttd_pur_df = dd.read_csv(f'{data_path}{file1_folder}TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
# abs_anttd_pur_df.head()
# -

# Convert to dask for distributed computing:

abs_anttd_pur_df = dd.from_pandas(abs_anttd_pur_df, npartitions=30)
abs_anttd_pur_df.head()

# #### Basic stats

abs_anttd_pur_df.dtypes

abs_anttd_pur_df.compute().nunique()

search_explore.dataframe_missing_values(abs_anttd_pur_df)

# Low percentages of missing values, topping at bellow 9%.

abs_anttd_pur_df = data_processing.standardize_missing_values_df(abs_anttd_pur_df)
abs_anttd_pur_df.head()

search_explore.dataframe_missing_values(abs_anttd_pur_df)

# In this dataset, all missing values were already well represented. We can see this as the missing values percentages didn't change after applying the `standardize_missing_values_df` method.

abs_anttd_pur_df.describe().compute().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

abs_anttd_pur_df['call status'].value_counts().compute()

# Not sure what this `call status` column represents.

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### Mutations data
#
# Description
# -

mut_df = pd.read_csv(f'{data_path}{mut_folder}mc3.v0.2.8.PUBLIC.maf.gz', compression='gzip', header=0, sep='\t')
# mut_df = pd.read_csv(f'{data_path}mc3.v0.2.8.PUBLIC.maf.gz', sep='\t')
mut_df.head()

len(list(mut_df.columns))

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# mut_df = dd.read_csv(f'{data_path}{file1_folder}mc3.v0.2.8.PUBLIC.maf.gz', sep='\t')
# mut_df.head()
# -

# Convert to dask for distributed computing:

mut_df = dd.from_pandas(mut_df, npartitions=60)
mut_df.head()

# **Comments:**
# * Some columns, such as `all_effects`, `Existing_variation`, `TREMBL` and `DOMAINS` seem to be lists of values, separated by commas (good candidates to use embedding bag).
# * The sample ID that we can use to match with the other tables appears to be `Tumor_Sample_Barcode`. There's also a similar `Matched_Norm_Sample_Barcode` column, but that seems to belong to another dataset (perhaps GTEx).
# * It looks like this table has a dot "." whenever it's a missing value.
# * There are several columns (114), with many of them not being clear as to what they represent nor if they're actually useful for this tumor type classification task.

# #### Basic stats

mut_df.dtypes

mut_df.compute().nunique()

search_explore.dataframe_missing_values(mut_df)

#

mut_df = data_processing.standardize_missing_values_df(mut_df)
mut_df.head()

search_explore.dataframe_missing_values(mut_df)

#

mut_df.describe().compute().transpose()

#

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### Clinical outcome (TCGA-CDR) data
#
# Description
# -

cdr_df = pd.read_excel(f'{data_path}{cdr_folder}TCGA-CDR-SupplementalTableS1.xlsx')
# cdr_df = pd.read_excel(f'{data_path}TCGA-CDR-SupplementalTableS1.xlsx')
cdr_df.head()

# +
# For some reason Dask is failing to read the excel file directly (failed to serialize)
# cdr_df = dd.read_excel(f'{data_path}{file1_folder}TCGA-CDR-SupplementalTableS1.xlsx')
# cdr_df.head()
# -

# Convert to dask for distributed computing:

cdr_df = dd.from_pandas(cdr_df, npartitions=30)
cdr_df.head()

# **Comments:**
# * Features such as `age_at_initial_pathologic_diagnosis`, `gender`, `race`, `ajcc_pathologic_tumor_stage`, `vital_status` and `tumor_status` might be very insteresting to include. Others such as overall survival (OS), progression-free interval (PFI), disease-free interval (DFI), and disease-specific survival (DSS) might not be relevant for this use case.
# * Missing values appear to be mostly represented as "[Not Applicable]", "[Not Available]", "[Not Evaluated]", "[Unknown]" or "[Discrepancy]".
# * Features related to outcomes, such as `treatment_outcome_first_course` and `death_days_to`, should be ignored, as we're classifying tumor type, regardless of the outcome.

# #### Basic stats

cdr_df.dtypes

cdr_df.compute().nunique()

search_explore.dataframe_missing_values(cdr_df)

# Most of the features with significant quantities of missing values (>40%) are not going to be used. But it's important to remember that categorical features like `clinical_stage` use different representations for missing values.

cdr_df.vital_status.value_counts().compute()

# Should probably assume the "[Discrepancy]" option as a missing value also.

cdr_df.ajcc_pathologic_tumor_stage.value_counts().compute()

cdr_df.clinical_stage.value_counts().compute()

# Considerable percentage of missing values on `ajcc_pathologic_tumor_stage` (\~37%) and `clinical_stage` (\~76%).

cdr_df = data_processing.standardize_missing_values_df(cdr_df)
cdr_df.head()

search_explore.dataframe_missing_values(cdr_df)

# Considering the real percentages of missing values, which are higher than what we got before standardizing the missing values representation, the main features to use from this table should be `gender`, `vital_status`, `age_at_initial_pathologic_diagnosis`, `tumor_status`, `race` and `ajcc_pathologic_tumor_stage`.

cdr_df.ajcc_pathologic_tumor_stage.value_counts().compute()

cdr_df.clinical_stage.value_counts().compute()

cdr_df.race.value_counts().compute()

cdr_df.histological_grade.value_counts().compute()

cdr_df.vital_status.value_counts().compute()

cdr_df.margin_status.value_counts().compute()

# Not sure what `margin_status` represents.

cdr_df.describe().compute().transpose()

# #### Tumor representation

cdr_df.type.value_counts().to_frame()

data = [go.Histogram(x = cdr_df.type)]
layout = go.Layout(title='Tumor types representation',
                   plot_bgcolor='#ffffff',
                   xaxis=dict(categoryorder='total descending'))
fig = go.Figure(data, layout)
py.iplot(fig)

# As expected, this table has similar tumor type representation as the RPPA table, although it has more samples.

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### Clinical with follow-up data
#
# Description
# -

clnc_fllw_df = pd.read_csv(f'{data_path}{clnc_fllw_folder}clinical_PANCAN_patient_with_followup.tsv', sep='\t', encoding='ISO-8859-1')
# clnc_fllw_df = pd.read_csv(f'{data_path}clinical_PANCAN_patient_with_followup.tsv', sep='\t')
clnc_fllw_df.head()

# +
# For some reason Dask is failing to read the CSV file directly (failed to serialize)
# clnc_fllw_df = dd.read_csv(f'{data_path}{file1_folder}clinical_PANCAN_patient_with_followup.tsv', sep='\t')
# clnc_fllw_df.head()
# -

# Convert to dask for distributed computing:

clnc_fllw_df = dd.from_pandas(clnc_fllw_df, npartitions=60)
clnc_fllw_df.head()

list(clnc_fllw_df.columns)

# **Comments:**
# * Appears to be similar to the TCGA-CDR table but with much more features. Most of these additional features seem to be irrelevant, however others like `height`, `weight`, `molecular_abnormality_results`, `prior_aids_conditions`, `hiv_status`, `on_haart_therapy_prior_to_cancer_diagnosis`, `tobacco_smoking_history`, `number_pack_years_smoked`, `history_immunological_disease_other`, `history_immunosuppressive_dx_other`, `history_immunological_disease`, `hemoglobin_result`, `white_cell_count_result`, `birth_control_pill_history_usage_category`, `alcohol_history_documented`, `frequency_of_alcohol_consumption`, `pregnancies`, `food_allergy_history`, `mental_status_changes`, `hay_fever_history`, and others could be interesting.

# #### Basic stats

clnc_fllw_df.dtypes

clnc_fllw_df.compute().nunique()

search_explore.dataframe_missing_values(clnc_fllw_df)

# The vast majority of the features are basically all filled with missing values (>80%). Despite features like `weight`, `height`, `molecular_abnormality_results`, `number_pack_years_smoked` and `tobacco_smoking_history` having less missing values and being potentially interesting, we still need to check what happens if we standardize the missing values representation, as the previous function only detects missing values when they're represented as NumPy's NaN value.

clnc_fllw_df.weight.value_counts().compute()

clnc_fllw_df.height.value_counts().compute()

clnc_fllw_df.molecular_abnormality_results.value_counts().compute()

clnc_fllw_df.number_pack_years_smoked.value_counts().compute()

clnc_fllw_df.tobacco_smoking_history.value_counts().compute()

clnc_fllw_df = data_processing.standardize_missing_values_df(clnc_fllw_df)
clnc_fllw_df.head()

search_explore.dataframe_missing_values(clnc_fllw_df)

# We can notice a general increase in the percentages of missing values, including on the features that we were interested on, namely `weight`, `height`, `molecular_abnormality_results`, `number_pack_years_smoked` and `tobacco_smoking_history`. Considering that they all have more than 70% missing values, they're likely of no use for us.

clnc_fllw_df.weight.value_counts().compute()

clnc_fllw_df.height.value_counts().compute()

clnc_fllw_df.molecular_abnormality_results.value_counts().compute()

clnc_fllw_df.number_pack_years_smoked.value_counts().compute()

clnc_fllw_df.tobacco_smoking_history.value_counts().compute()

clnc_fllw_df.describe().compute().transpose()

# Considering that this table is essentially an extension of the TCHA-CDR table, with more features but with all of them having large amounts of missing values (>70%), we're not going to use this table.
