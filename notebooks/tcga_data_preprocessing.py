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

# # TCGA Data Preprocessing
# ---
#
# Preprocessing the TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048).
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import os                                  # os handles directory/workspace changes
import yaml                                # Save and load YAML files
import numpy as np                         # NumPy to handle numeric and NaN operations
from functools import reduce               # Parallelize functions
# -

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'data/TCGA-Pancancer/'
rppa_folder = 'original/fcbb373e-28d4-4818-92f3-601ede3da5e1/'
dna_mthltn_folder = 'original/d82e2c44-89eb-43d9-b6d3-712732bf6a53/'
abs_anttd_pur_folder = 'original/4f277128-f793-4354-a13d-30cc7fe9f6b5/'
rna_folder = 'original/3586c0da-64d0-4b74-a449-5ff4d9136611/'
mut_folder = 'original/1c8cfe5f-e52d-41ba-94da-f15ea1337efc/'
mirna_folder = 'original/1c6174d9-8ffb-466e-b5ee-07b204c15cf8/'
cdr_folder = 'original/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/'
clnc_fllw_folder = 'original/0fc78496-818b-4896-bd83-52db1f533c5c/'
abs_anttd_seg_folder = 'original/0f4f5701-7b61-41ae-bda9-2805d1ca9781/'

import modin.pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# + [markdown] {"toc-hr-collapsed": true}
# ## RPPA data
#
# Reverse phase protein array (RPPA) is a high-throughput antibody-based technique with the procedures similar to that of Western blots. Proteins are extracted from tumor tissue or cultured cells, denatured by SDS , printed on nitrocellulose-coated slides followed by antibody probe. Our RPPA platform currently allows for the analysis of >1000 samples using at least 130 different antibodies.
# -

# ### Loading the data

rppa_df = pd.read_csv(f'{data_path}{rppa_folder}TCGA-RPPA-pancan-clean.csv')
rppa_df.head()

# ### Setting the index

# Set `sample_id` column to be the index:

rppa_df = rppa_df.set_index('SampleID')
rppa_df.head()

# Fix the index name:

rppa_df = du.data_processing.rename_index(rppa_df, 'sample_id')
rppa_df.head()

# ### Checking for missing values

du.search_explore.dataframe_missing_values(rppa_df)

# Out of all the 200 columns, only 9 of them have missing values, with 8 having more than 49% (`ARID1A`, `ADAR1`, `ALPHACATENIN`, `TTF1`, `PARP1`, `JAB1`, `CASPASE9`, `CASPASE3`).

# ### Removing unneeded features

# Remove columns that have more than 40% missing values:

rppa_df = du.data_processing.remove_cols_with_many_nans(rppa_df, nan_percent_thrsh=40, inplace=True)

du.search_explore.dataframe_missing_values(rppa_df)

# ### Normalizing data

rppa_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# Save the dataframe before normalizing:

rppa_df.to_csv(f'{data_path}cleaned/unnormalized/rppa.csv')

# Normalize the data into a new dataframe:

rppa_df_norm = du.data_processing.normalize_data(rppa_df, id_columns=None)
rppa_df_norm.head()

# Confirm that everything is ok through the `describe` method:

rppa_df_norm.describe().transpose()

# Save the normalized dataframe:

rppa_df_norm.to_csv(f'{data_path}cleaned/normalized/rppa.csv')

# + [markdown] {"toc-hr-collapsed": true}
# ## RNA data
#
# Description
# -

# ### Loading the data

rna_df = pd.read_csv(f'{data_path}{rna_folder}EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep='\t')
rna_df.head()

# ### Setting the index

# This dataframe is inverted, i.e. the columns should switch with the rows.

rna_df = du.data_processing.transpose_dataframe(rna_df, column_to_transpose='gene_id', inplace=True)
rna_df.head()

# Fix the index name:

rna_df = du.data_processing.rename_index(rna_df, 'sample_id')
rna_df.head()

# ### Checking for missing values

du.search_explore.dataframe_missing_values(rna_df)

# No gene has more than 16% missing values.

# ### Normalizing data

rna_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# Save the dataframe before normalizing:

rna_df.to_csv(f'{data_path}cleaned/unnormalized/rna.csv')

# Normalize the data into a new dataframe:

rna_df_norm = du.data_processing.normalize_data(rna_df, id_columns=None)
rna_df_norm.head()

# Confirm that everything is ok through the `describe` method:

rna_df_norm.describe().transpose()

# Save the normalized dataframe:

rna_df_norm.to_csv(f'{data_path}cleaned/normalized/rna.csv')

rna_df_norm.head()

# + [markdown] {"toc-hr-collapsed": true}
# ## DNA Methylation
#
# Description
# -

# ### Loading the data

dna_mthltn_df = pd.read_csv(f'{data_path}{dna_mthltn_folder}jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv', sep='\t')
dna_mthltn_df.head()

# ### Setting the index

# This dataframe is inverted, i.e. the columns should switch with the rows.

dna_mthltn_df = du.data_processing.transpose_dataframe(dna_mthltn_df, column_to_transpose='Composite Element REF', inplace=True)
dna_mthltn_df.head()

# Fix the index name:

dna_mthltn_df = du.data_processing.rename_index(dna_mthltn_df, 'sample_id')
dna_mthltn_df.head()

# ### Checking for missing values

du.search_explore.dataframe_missing_values(dna_mthltn_df)

# The DNA composite with the most missing values only has less than 35% missingness; 
# However, although it seems like this table has the right missing values representation, 
# we haven't done missing values standardization.

# ### Removing unneeded features

# Remove columns that have more than 40% missing values:

dna_mthltn_df = du.data_processing.remove_cols_with_many_nans(dna_mthltn_df, nan_percent_thrsh=40, inplace=True)

du.search_explore.dataframe_missing_values(dna_mthltn_df)

# ### Normalizing data

dna_mthltn_df.describe().transpose()

# Save the dataframe before normalizing:

dna_mthltn_df.to_csv(f'{data_path}cleaned/unnormalized/dna_methylation.csv')

# Normalize the data into a new dataframe:

dna_mthltn_df_norm = du.data_processing.normalize_data(dna_mthltn_df, id_columns=None)
dna_mthltn_df_norm.head()

# Confirm that everything is ok through the `describe` method:

dna_mthltn_df_norm.describe().transpose()

# Save the normalized dataframe:

dna_mthltn_df_norm.to_csv(f'{data_path}cleaned/normalized/dna_methylation.csv')

# + [markdown] {"toc-hr-collapsed": true}
# ## miRNA data
#
# Description
# -

# ### Loading the data

mirna_df = pd.read_csv(f'{data_path}{mirna_folder}pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv')
mirna_df.head()

# ### Removing uncorrected genes

mirna_df.Correction.value_counts()

# Since only 81 genes are "uncorrected" (probably means no preprocessing, such as removing batch effects, was done), we should consider removing them;
# For now, we'll simply drop the `Correction` column.

mirna_df = mirna_df[mirna_df['Correction'] == 'Corrected']
mirna_df.head()

mirna_df.Correction.value_counts()

mirna_df = mirna_df.drop(columns='Correction')
mirna_df.head()

# ### Setting the index

# This dataframe is inverted, i.e. the columns should switch with the rows.

mirna_df = du.data_processing.transpose_dataframe(mirna_df, column_to_transpose='Genes', inplace=True)
mirna_df.head()

# Fix the index name:

mirna_df = du.data_processing.rename_index(mirna_df, 'sample_id')
mirna_df.head()

# ### Checking for missing values

du.search_explore.dataframe_missing_values(mirna_df)

# Absolutely no missing values in this dataframe!

# ### Normalizing data

mirna_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# Save the dataframe before normalizing:

mirna_df.to_csv(f'{data_path}cleaned/unnormalized/mirna.csv')

# Normalize the data into a new dataframe:

mirna_df_norm = du.data_processing.normalize_data(mirna_df, id_columns=None)
mirna_df_norm.head()

# Confirm that everything is ok through the `describe` method:

mirna_df_norm.describe().transpose()

# Save the normalized dataframe:

mirna_df_norm.to_csv(f'{data_path}cleaned/normalized/mirna.csv')

# + [markdown] {"toc-hr-collapsed": false}
# ## ABSOLUTE-annotated seg data
#
# This dataframe contains copy-number and copy-ratio related data.
#
# Copy number alterations/aberrations (CNAs) are changes in copy number that have arisen in somatic tissue (for example, just in a tumor), copy number variations (CNVs) originated from changes in copy number in germline cells (and are thus in all cells of the organism).
#
# The rows correspond to contiguous chunks along the chromosome with the same DNA copy-number. "Chromosome" is the chromosome, can be 1-22, X or Y (see human genome). Start is the physical start location for the segment along said linear chromosome, end is the end coordinate. Num_probes is the number of SNP-array probes falling within the segment (these were used to call copy numbers). Reference: https://www.biostars.org/p/244374/
# -

# ### Loading the data

abs_anttd_seg_df = pd.read_csv(f'{data_path}{abs_anttd_seg_folder}TCGA_mastercalls.abs_segtabs.fixed.txt', sep='\t')
abs_anttd_seg_df.head()

abs_anttd_seg_df.Sample.nunique()

len(abs_anttd_seg_df)

# ### Checking for missing values

du.search_explore.dataframe_missing_values(abs_anttd_seg_df)

# Low percentages of missing values, topping at bellow 8%.

# ### Converting categorical features to numeric

abs_anttd_seg_df.solution.value_counts()

abs_anttd_seg_df.solution = abs_anttd_seg_df.solution.apply(lambda x: 1 if x == 'new' else 0)
abs_anttd_seg_df = abs_anttd_seg_df.rename(columns={'solution': 'new_solution'})
abs_anttd_seg_df.new_solution.value_counts()

# ### Removing unneeded features

# Columns `Start`, `End`, `Num_Probes` and `Length` don't seem to be relevant as we don't need to know so much detail of each chromosome nor experiment specific information.

abs_anttd_seg_df = abs_anttd_seg_df.drop(columns=['Start', 'End', 'Num_Probes', 'Length'], axis=1)
abs_anttd_seg_df.head()

# ### Normalizing data

abs_anttd_seg_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# Save the dataframe before normalizing:

abs_anttd_seg_df.to_csv(f'{data_path}cleaned/unnormalized/copy_number_ratio.csv')

# Normalize the data into a new dataframe:

abs_anttd_seg_df_norm = du.data_processing.normalize_data(abs_anttd_seg_df, id_columns=None, categ_columns='Chromosome')
abs_anttd_seg_df_norm.head()

# Confirm that everything is ok through the `describe` method:

abs_anttd_seg_df_norm.describe().transpose()

# + [markdown] {"toc-hr-collapsed": false}
# ### Aggregating sample data
# -

# #### Missing value imputation
#
# We can't join rows correctly if there are missing values

nan_idx = abs_anttd_seg_df[abs_anttd_seg_df.Ccf_ci95_high_a2.isnull()].index
nan_idx

abs_anttd_seg_df.iloc[nan_idx].head()

abs_anttd_seg_df.head(125).tail(25)

abs_anttd_seg_df = du.data_processing.missing_values_imputation(abs_anttd_seg_df, method='interpolation',
                                                                id_column='Sample', inplace=True)
abs_anttd_seg_df.head()

du.search_explore.dataframe_missing_values(abs_anttd_seg_df)

abs_anttd_seg_df.head(125).tail(25)

# #### Average groupby aggregation
#
# Join all the data of each sample's chromosome through an average groupby aggregation:

abs_anttd_seg_df = abs_anttd_seg_df.groupby(['Sample', 'Chromosome']).mean()
abs_anttd_seg_df.head(25)

# #### Dividing chromosome data into different columns
#
# Separate each chromosome's information into their own features.
#
# OR
#
# Create lists for each feature, containing each chromosome's value, and then apply an embedding bag on it.

abs_anttd_seg_df[abs_anttd_seg_df.index.get_level_values('Chromosome') == 1].head()

# List that will contain multiple dataframes, one for each chromosome
df_list = []
# Go through each chromosome and create its own dataframe, with properly labeled columns
for chrom in range(1, 23):
    # Filter for the current chromosome's dataframe
    tmp_df = abs_anttd_seg_df[abs_anttd_seg_df.index.get_level_values('Chromosome') == chrom]
    # Change the column names to identify the chromosome
    tmp_df.columns = [f'{col}_chromosome_{chrom}' for col in tmp_df.columns]
    # Remove now redundant `Chromosome` column
    tmp_df = tmp_df.reset_index().drop(columns='Chromosome', axis=1)
    # Add to the dataframes list
    df_list.append(tmp_df)

df_list[3]

abs_anttd_seg_df = reduce(lambda x, y: pd.merge(x, y, on='Sample'), df_list)
abs_anttd_seg_df.head()

abs_anttd_seg_df.Sample.nunique()

len(abs_anttd_seg_df)

# +
# [TODO] See if there are duplicate columns; I suspect that at least some binary columns, like new_solution, are the same for every chromosome
# -

# ### Setting the index

# Set `sample_id` column to be the index:

abs_anttd_seg_df = abs_anttd_seg_df.set_index('Sample')
abs_anttd_seg_df.head()

# Fix the index name:

abs_anttd_seg_df = du.data_processing.rename_index(abs_anttd_seg_df, 'sample_id')
abs_anttd_seg_df.head()

# Save the normalized dataframe:

abs_anttd_seg_df_norm.to_csv(f'{data_path}cleaned/normalized/copy_number_ratio.csv')

# + [markdown] {"toc-hr-collapsed": true}
# ## ABSOLUTE purity/ploidy data
#
# Description
# -

# ### Loading the data

abs_anttd_pur_df = pd.read_csv(f'{data_path}{abs_anttd_pur_folder}TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
abs_anttd_pur_df.head()

# ### Setting the index

# Set `sample_id` column to be the index:

abs_anttd_pur_df = abs_anttd_pur_df.set_index('sample')
abs_anttd_pur_df.head()

# Fix the index name:

abs_anttd_pur_df = du.data_processing.rename_index(abs_anttd_pur_df, 'sample_id')
abs_anttd_pur_df.head()

# ### Checking for missing values

du.search_explore.dataframe_missing_values(abs_anttd_pur_df)

# Low percentages of missing values, topping at bellow 9%.

# ### Converting categorical features to numeric

abs_anttd_pur_df.solution.value_counts()

abs_anttd_pur_df.solution = abs_anttd_pur_df.solution.apply(lambda x: 1 if x == 'new' else 0)
abs_anttd_pur_df = abs_anttd_pur_df.rename(columns={'solution': 'new_solution'})
abs_anttd_pur_df.new_solution.value_counts()

# ### Removing unneeded features

abs_anttd_pur_df['array'].value_counts()

abs_anttd_pur_df['call status'].value_counts()

# We're going to remove the redundant `array` (contains a less detailed version of the `sample_id`) and the apparently irrelevant `call status`:

abs_anttd_pur_df = abs_anttd_pur_df.drop(columns=['array', 'call status'], axis=1)
abs_anttd_pur_df.head()

# ### Normalizing data
#
# [TODO] Consider not removing the fraction data columns, namely `purity`, `Cancer DNA fraction` and `Subclonal genome fraction`. The only issue is how do I do imputation then, if 0 doesn't necessarily correspond to the average value.

abs_anttd_pur_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# Save the dataframe before normalizing:

abs_anttd_pur_df.to_csv(f'{data_path}cleaned/unnormalized/purity_ploidy.csv')

# Normalize the data into a new dataframe:

abs_anttd_pur_df_norm = du.data_processing.normalize_data(abs_anttd_pur_df, id_columns=None)
abs_anttd_pur_df_norm.head()

# Confirm that everything is ok through the `describe` method:

abs_anttd_pur_df_norm.describe().transpose()

# Save the normalized dataframe:

abs_anttd_pur_df_norm.to_csv(f'{data_path}cleaned/normalized/purity_ploidy.csv')

# + [markdown] {"toc-hr-collapsed": true}
# ## Mutations data
#
# Description
#
# [TODO] Ignoring mutation data for now, for the confusing, unstructured mess that it is. Consider adding it later.
# -

# ### Loading the data

mut_df = pd.read_csv(f'{data_path}{mut_folder}mc3.v0.2.8.PUBLIC.maf.gz',
                     compression='gzip', header=0, sep='\t')
mut_df.head()

mut_df.dtypes

# ### Setting the index

# Set `sample_id` column to be the index:

mut_df = mut_df.set_index('sample_id')
mut_df.head()

# Fix the index name:

mut_df = du.data_processing.rename_index(mut_df, 'sample_id')
mut_df.head()

# ### Checking for missing values

du.search_explore.dataframe_missing_values(mut_df)

# Out of all the 200 columns, only 9 of them have missing values, with 8 having more than 49% (`ARID1A`, `ADAR1`, `ALPHACATENIN`, `TTF1`, `PARP1`, `JAB1`, `CASPASE9`, `CASPASE3`).

# ### Removing unneeded features

# Remove columns that have more than 40% missing values:

mut_df = du.data_processing.remove_cols_with_many_nans(mut_df, nan_percent_thrsh=40, inplace=True)

du.search_explore.dataframe_missing_values(mut_df)

# **Comments:**
# * Some columns, such as `all_effects`, `Existing_variation`, `TREMBL` and `DOMAINS` seem to be lists of values, separated by commas (good candidates to use embedding bag).
# * The sample ID that we can use to match with the other tables appears to be `Tumor_Sample_Barcode`. There's also a similar `Matched_Norm_Sample_Barcode` column, but that seems to belong to another dataset (perhaps GTEx).
# * It looks like this table has a dot "." whenever it's a missing value.
# * There are several columns (114), with many of them not being clear as to what they represent nor if they're actually useful for this tumor type classification task.

# ### Normalizing data

mut_df.describe().transpose()

# The data is not (well) normalized yet. All columns should have 0 mean and 1 standard deviation.

# Save the dataframe before normalizing:

mut_df.to_csv(f'{data_path}cleaned/unnormalized/mutation.csv')

# Normalize the data into a new dataframe:

mut_df_norm = du.data_processing.normalize_data(mut_df, id_columns=None)
mut_df_norm.head()

# Confirm that everything is ok through the `describe` method:

mut_df_norm.describe().transpose()

# Save the normalized dataframe:

mut_df_norm.to_csv(f'{data_path}cleaned/normalized/mutation.csv')

# + [markdown] {"toc-hr-collapsed": true}
# ## Clinical outcome (TCGA-CDR) data
#
# Description
# -

# ### Loading the data

cdr_df = pd.read_excel(f'{data_path}{cdr_folder}TCGA-CDR-SupplementalTableS1.xlsx')
cdr_df.head()

cdr_df.dtypes

# ### Setting the index

# Set `sample_id` column to be the index:

cdr_df = cdr_df.set_index('bcr_patient_barcode')
cdr_df.head()

# Fix the index name:

cdr_df = du.data_processing.rename_index(cdr_df, 'sample_id')
cdr_df.head()

# ### Checking for missing values

du.search_explore.dataframe_missing_values(cdr_df)

cdr_df = du.data_processing.standardize_missing_values_df(cdr_df)
cdr_df.head()

du.search_explore.dataframe_missing_values(cdr_df)

# Considerable percentage of missing values on `ajcc_pathologic_tumor_stage` (\~37%) and `clinical_stage` (\~76%).
# Considering the real percentages of missing values, which are higher than what we got before standardizing the missing values representation, the main features to use from this table should be `gender`, `vital_status`, `age_at_initial_pathologic_diagnosis`, `tumor_status`, `race` and `ajcc_pathologic_tumor_stage`.

# ### Removing unneeded features

# Remove columns that have more than 40% missing values:

cdr_df = du.data_processing.remove_cols_with_many_nans(cdr_df, nan_percent_thrsh=40, inplace=True)

du.search_explore.dataframe_missing_values(cdr_df)

# Features such as overall survival (`OS`), progression-free interval (`PFI`), disease-specific survival (`DSS`), `vital_status`, `tumor_status`, `initial_pathologic_dx_year`, `birth_days_to` and `last_contact_days_to`,  might not be relevant for this use case. Also, both `type` and `histological_type` already indicate what tumor type it is, which is our intended label, so we must remove them.

cdr_df = cdr_df.drop(columns=['Unnamed: 0', 'OS', 'PFI', 'DSS',
                              'OS.time', 'DSS.time', 'PFI.time',
                              'vital_status', 'tumor_status', 
                              'initial_pathologic_dx_year', 'birth_days_to',
                              'last_contact_days_to', 'type',
                              'histological_type'], axis=1)
cdr_df.head()

# ### Converting categorical features to numeric

cdr_df.gender.value_counts()

cdr_df.race.value_counts()

cdr_df.ajcc_pathologic_tumor_stage.value_counts()

# Encode gender:

cdr_df.gender = cdr_df.gender.apply(lambda x: 1 if x.lower() == 'male' else 0)
cdr_df.gender.value_counts()

# Encode race and tumor stage:

features_to_encode = ['race', 'ajcc_pathologic_tumor_stage']

# Dictionary that will contain the mapping between the categories and their encodings
encod_dict = dict([('gender', dict([('male' , 1), ('female', 0)]))])

for feature in features_to_encode:
    cdr_df[feature], encod_dict[feature] = du.embedding.enum_categorical_feature(cdr_df, feature)

cdr_df.head()

encod_dict

# ### Normalizing data
#
# In this table, we only need to normalize the age.

cdr_df.describe().transpose()

# Save the dataframe before normalizing:

cdr_df.to_csv(f'{data_path}cleaned/unnormalized/clinical_outcome.csv')

# Normalize the data into a new dataframe:

cdr_df_norm = du.data_processing.normalize_data(cdr_df, id_columns=None, columns_to_normalize='age_at_initial_pathologic_diagnosis')
cdr_df_norm.head()

# Confirm that everything is ok through the `describe` method:

cdr_df_norm.describe().transpose()

# Save the normalized dataframe:

cdr_df_norm.to_csv(f'{data_path}cleaned/normalized/clinical_outcome.csv')

# ## Saving enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

stream = open(f'{data_path}cleaned/encod_dict.yaml', 'w')
yaml.dump(encod_dict, stream, default_flow_style=False)


