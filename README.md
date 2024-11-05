# Requirements Classification Pipeline for Requirements Reuse

In various domains, e.g., aero-space or automotive, standards are applied to reuse requirements and ensure a high level of product quality and safety. During **standard tailoring**, requirements from the applicable standards are specialized and integrated into the project. However, the requirement type influences the way the standard requirement interacts with project requirements. There are five different types: *functional* (F), *non-functional* (NF), *project management* (PM), *meta* (M), and *view* (V) requirements. To support the process of standard tailoring, the requirements from the applicable standards should be classified before integration. Yet, manual classification of large existing standards is time-consuming. This artifact presents a **classification pipeline** to compare **five machine learning algorithms** for this task: *k-Nearest Neighbour* (kNN), *Support Vector Machine* (SVM), *Logistic Regression* (LR), *Multinomial Naive Bayes* (MNB), *Random Forest* (RF), as well as an *ensemble model* combining all five. A set of classified requirements serves as input and, after various preparation steps, is used to train and test the models created by the different algorithms. The evaluation results, showing how well the algorithms perform, are the output of the pipeline.

Further, we propose an approach to extend the classification by the context of terms contained in a requirement to potentially improve the performance. The contextualization is based on the hierarchy levels of the ECSS and integrated in the *Preprocessing* phase of the pipeline.

The six phases of the classification pipeline:
![pipeline image](resources/pipeline_image.jpg?raw=true "Requirements Classification Pipeline")

## Description of Artifact
The models were originally trained and tested with 466 requirements from the European Cooperation for Space Standardisation [(ECSS)](https://ecss.nl/). However, due to IP rights of ECSS (https://ecss.nl/license-agreement-disclaimer/) the full dataset cannot be republished (request for permission pending). Thus, we provide the [PROMISE_exp](http://ctp.di.fct.unl.pt/RE2017/downloads/datasets/nfr.arff)requirements set as a placeholder input dataset so that the processing stages of the pipline can be understood despite the missing set of standard requirements. But note that the PROMISE_exp dataset has been artificially extended by us so that we can demonstrate our approach for context integration. In the extended version, ECSS standard IDs serve as a dummy requirement source reference. However, there is no real connection between ECSS standards and the requirements of the PROMISE_exp dataset.

The artifact is structured as follows:

### Evaluation Results for ECSS Data
The folder 'evaluation_results_for_ECSS_data' contains the evaluation results obtained from the original ECSS data. This includes the evaluation results of the classical pipeline (subfolder 'without_context_integration') as well as the results of the extended pipeline (subfolder 'with_context_integration'). Both subfolders contain a table with the precision, recall, and F1 score of the algorithms ('evaluation_metrics.xlsx'), tables with the standard deviations measured during cross-validation ('evaluation_precision_standard_deviation.xlsx', 'evaluation_recall_standard_deviation.xlsx', 'evaluation_f1_standard_deviation.xlsx'), a table showing the three metrics per label for every algorithm ('evaluation_labels.xlsx'), a column chart visualizing the achieved F1 scores ('evaluation_f1_column_chart.xlsx'), the mean times for training and testing the models ('evaluation_time.xlsx'), and an analysis of variance (ANOVA) to prove (in-)significance of performance differences ('evaluation_analysis.xlsx').

### Resources
The folder 'resources' initially contained the requirements data basis 'EARM_ECSS_export(DOORS-v0.9_v2_21Feb2023).xlsx' retrieved from the [ECSS DOORS database v.0.9](https://ecss.nl/standards/downloads/earm/). This file is empty due to IP rights of ECSS but should indicate our original procedure. 'ECSS_standards_classified.csv' can be used for reproduction (see [Steps to Reproduce](#steps-to-reproduce)).

The subfolder 'ECSS_term_contexts' contains all data relevant for the integration of term contexts. 'ECSS-Abbreviated-Terms_active-and-superseded-Standards-(from-ECSS-DOORS-database-v0.9_5Oct2022).xlsx' and 'ECSS-Definitions_active-and-superseded-Standards-(from-ECSS-DOORS-database-v0.9_5Oct2022).xlsx' are again empty files originally retrieved from the [ECSS DOORS database v.0.9](https://ecss.nl/glossary/glossary-definitions-abbreviated-terms/). They indicate where the information on ECSS terms and abbreviations initially comes from and are used by 'lookup_tables_builder.py' to create the lookup tables 'lookup_terms.csv' and 'lookup_abbreviations.csv'. They assign full terms and abbreviations of the ECSS to the standards and branches in which they are defined.

### Data Collection
The folder '0_data_collection' corresponds to the first phase of the pipeline where the dataset for training and testing is build. 'data_collection.py' collects all classified requirements from 'EARM_ECSS_export(DOORS-v0.9_v2_21Feb2023).xlsx' and stores the prepared requirements set in the 'output' folder where in this case the extended PROMISE_exp dataset lies.

### Preprocessing
The folder '1_preprocessing' corresponds to the second phase. Running 'preprocessing.py' means that the collected requirements are cleaned and normalized. The resulting requirements set is stored in 'output'.
Additionally, in this phase the optional integration of term contexts can be carried out by running 'context_integration.py'. The mere requirement texts are complemented by pairs of terms and the associated contexts. The extended data is also stored in 'output' to be passed on to the next phase.

### Feature Extraction
The folder '2_feature_extraction' corresponds to the third phase where the (extended) requirement texts are transformed into feature vectors using *Bag of Words* (BoW) and *Term Frequency - Invers Document Frequency* (TF-IDF). By running 'feature_extraction.py' the preprocessed requirements data (with and without context extension) is vectorized and stored in 'output'.

### Dimensionality Reduction
The folder '3_dimensionality_reduction' corresponds to the fourth phase where the dimensionality of the feature vectors created in the previous phase is reduced by using *feature selection with chi^2* ('feature_selection.py') and *Principal Component Analysis* (PCA) ('principal_component_analysis.py'). Both techniques can be applied by running the respective .py files. Note that 'feature_selection.py' takes the data from the previous phase as input while 'principal_component_analysis.py' further reduces the data generated by feature selection. To compare both variants (only feature selection and feature selection with PCA), there are four subfolders in 'output' holding the respective variants of preparation for every algorithm.
Further, the subfolder 'hyperparameter_tuning' contains 'hyperparameter_tuning.py', which can be run to find the best hyperparameter settings for the different algorithms. The settings determined for the original ECSS data can be found in 'hyperparameter_tuning_results.txt'.

### Classification & Evaluation
The folder '4_classification&evaluation' corresponds to the last two phases of the pipeline where the models are created and evaluated by the metrics *precision*, *recall*, and *F1 score*. The metrics are calculated by cross-validation to ensure reliable results. To obtain the evaluation results for the requirements data not extended by term context, one can run 'classification.py' and for the data extended by term context 'classification_context.py'. All evaluation results are then stored in 'output', which is again divided into two folders for the data with and without context integration. The evaluation results include the metric values for all algorithms and variants of data preparation and more (see [Evaluation Results for ECSS Data](#evaluation-results-for-ecss-data)). They make it possible to compare the machine learning models regarding significance of performance differences (especially interesting for our approach of term context integration).

## System Requirements
To run the classification pipeline, your system must meet the following requirements:
- On Windows:
    - Python 3.11.9
    - Pip 23.3.2
    - IDE suitable for Python (e.g. Visual Studio Code) 
    - For .xlsx data files: Microsoft Excel 2016 or newer
    - For .csv files: text or spreadsheet editor

-On Linux:


## Installation Instructions
To run the artifact, follow these steps:

### 1) Clone the repository into your IDE. 
### 2) Create a virtual environment for the project:
```
python -m venv .venv
```
### 3) Activate the virtual environment:
```
.venv\Scripts\activate
```
### 4) Install the required libraries:
```
pip install -r requirements.txt
```
Now, you are able to run all Python files in this repository.

## Usage Instructions
Explain (preferably with a running example) how the artifact can be used
For automated analyses or tools, this should include instructions on how to interact with the tool, API documentation, and all the information that enables other subjects to reuse the artifact

The pipeline is essentially realized by the five folders: 
- '0_data_collection'
- '1_preprocessing'
- '2_feature_extraction'
- '3_dimensionality_reduction'
- '4_classification&evaluation'

A requirements input set collected during *data collection* and which is contained in the 'output' folder of this phase (in our case 'PROMISE_dummy_Standards.csv') can, step by step, be sent through the classification pipline. To do so, one manually runs the respective .py files of the listed directories as detailedly explained in [Description of Artifact](#description-of-artifact). By that, the processing steps of the different phases can be retraced and one can view the data which is passed from phase to phase in the respective 'output' folders.

## Steps to Reproduce
As already mentioned, the originally used ECSS input dataset can not be republished, but we can give instructions how to reproduce it. 'ECSS_standards_classified.csv' in the 'resource' folder provides a list of the requirement IDs, their source standards, and our manually assigned classes. After regisration at https://ecss.nl/register/, one can access the respective standards and reproduce the original dataset.
If one then runs the pipeline with the ECSS dataset, they can obtain the results presented in the paper apart from values produced by RF. They can differ even with the same input due to the random factor within RF.

## Authors Information
University of Koblenz, Institute for Software Technology (IST):
- Julia Märdian (maerdian01@uni-koblenz.de)
- Katharina Großer (grosser@uni-koblenz.de)
- Jan Jürjens (juerjens@uni-koblenz.de)

## Artifact Location
