# Requirements Classification Pipeline for Requirements Reuse

In various domains, e.g., aero-space or automotive, standards are applied to reuse requirements and ensure a high level of product quality and safety. During **standard tailoring**, requirements from the applicable standards are specialized and integrated into the project. However, the requirement type influences the way the standard requirement interacts with project requirements. There are five different types: *functional* (F), *non-functional* (NF), *project management* (PM), *meta* (M), and *view* (V) requirements. To support the process of standard tailoring, the requirements from the applicable standards should be classified before integration. Yet, manual classification of large existing standards is time-consuming. This artifact presents a **classification pipeline** to compare five **machine learning algorithms** for this task: *k-Nearest Neighbour* (kNN), *Support Vector Machine* (SVM), *Logistic Regression* (LR), *Multinomial Naive Bayes* (MNB), *Random Forest* (RF), as well as an *ensemble model* combining all five. A set of classified requirements serves as input and, after various preparation steps, is used to train and test the models created by the different algorithms. The evaluation results, showing how well the algorithms perform, are the output of the pipeline.

## Description of Artifact
The models are originally trained and tested with 466 requirements from the European Cooperation for Space Standardisation~(ECSS).
Unfortunately,

## System Requirements

virtual environment

```
pip install -r requirements.txt
```
## Installation Instructions

## Usage Instructions

## Steps to Reproduce

## Authors Information

## Artifact Location
