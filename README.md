# tcga-cancer-classification
A deep learning approach to classify cancer type from TCGA data.

## Reproducing the results

While the data must be download from the original source, all the code needed to reproduce the paper's results is in this repository. You should install the required dependencies, either through [Poetry](https://poetry.eustace.io/) (recommended) or the `requirements.txt` file, and use the original random seed that's defined in each notebook. As the dataset and the models are somewhat memory heavy (I've experienced the use of around 30GB of RAM), make sure that you have a computer with enough RAM memory. In case you don't have a powerful enough personal computer, I recommend that you use the [Google AI Platform](https://cloud.google.com/ai-platform-notebooks/). Then, you just need to follow these next steps:

**1. Download the data**

Head to the [Pan-Cancer Atlas page](https://gdc.cancer.gov/about-data/publications/pancanatlas) and download all the supplemental data. It has already gone through some preprocessing, such as reducing batch effects. In the this project, we end up only using RNA, copy number (ABSOLUTE-annotated seg file) and clinical outcome (CDR) data, but you can download the remaining ones if you want to experiment with them (particularly in the data exploration and preprocessing notebooks).
    
**2. Explore the data**

Go through each file and explore what's inside, getting a first glimpse into the features and the dataset stats. This is done through the `tcga_data_exploration` notebook. Feel free to add more experiments of your own, after cloning the repository.

**3. Preprocess the data**

Prepare the data for the tumor classification task by preprocessing it in the `tcga_data_preprocessing` notebook. Afterwards, join the dataframes into one in the `tcga_data_joining` notebook.

**4. Train the machine learning models**

With the data already preprocessed, it's time to train the machine learning models on it. You'll train support vector machine (SVM), logistic regression, XGBoost and multilayer perceptron (mlp) models in the `tcga-model-training` notebook.

**5. Interpret the models**

Using [SHAP](https://github.com/slundberg/shap), you can now interpret the already trained models, analysing feature importance for each one. Open the `tcga-model-interpretation` notebook to do this. Be aware that the calculations for the MLP and the SVM models can take a lot of time (around 5 hours in my virtual machine that had 16 vCPUs).
