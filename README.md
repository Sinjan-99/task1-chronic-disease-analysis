# Chronic disease analytics — Individual Task 1, Part 1.3

Analysis supporting a Case Studies in Data Science assignment (COSC2669, RMIT University). Two publicly available healthcare datasets, two classification algorithms each, framed around a Data Analyst role at a primary health network.

Datasets

Both from the UCI Machine Learning Repository, downloaded at runtime.

Dataset	Size	Target
Diabetes 130-US Hospitals, 1999–2008	101,766 encounters, 50 attributes	Readmission within 30 days (~11% positive)
Heart Disease, Cleveland	303 patients, 14 attributes	Angiographic disease presence (~46% positive)
Models

Regularised logistic regression with balanced class weighting, and histogram-based gradient boosting. Logistic regression gives signed coefficients that can be inspected directly; gradient boosting picks up interactions a linear model cannot represent.

Running it
bash
pip install ucimlrepo scikit-learn pandas matplotlib
jupyter notebook Task1_Part1_3.ipynb

First run downloads both datasets from UCI and caches them to data/, so later runs work offline. Figures and CSV results are written to figures/. Works in Google Colab without changes.

Preprocessing decisions

The diabetes data needs several judgement calls, all made in section 2 of the notebook:

Missing values are encoded ? rather than as nulls.
weight is dropped at roughly 97% missing — imputation at that rate would be fabrication rather than recovery.
Encounters with discharge dispositions indicating death or hospice transfer are removed, since those patients cannot be readmitted.
Records are reduced to one encounter per patient. Without this the same individual appears in both the training and test partitions.
ICD-9 diagnosis codes are grouped into nine clinical chapters, following Strack et al. (2014), rather than one-hot encoding 700+ raw codes.
Evaluation

Accuracy is rejected as the headline metric for the diabetes data: at ~11% prevalence, predicting no readmission for every patient scores ~89% while identifying nobody. Average precision is used instead, following Saito and Rehmsmeier (2015).

The notebook also reports precision within the top decile of predicted risk, which is the operationally meaningful number when a care-coordination team has fixed contact capacity, plus a threshold sensitivity table so the operating point can be chosen against the real cost asymmetry rather than left at 0.5.

The Cleveland data is roughly balanced, so accuracy is interpretable, but n = 303 makes a single split unstable — results are reported as means and standard deviations across stratified 10-fold cross-validation.

A note on the loader

ucimlrepo returns a dataset in three blocks: features, targets, and sometimes ids. The diabetes data puts encounter_id and patient_nbr in ids. An earlier version joined only features and targets, which dropped patient_nbr and silently disabled the deduplication step above — the row count after cleaning was 99,343 against the 71,518 unique patients the dataset contains. All three blocks are now joined at load time, and assertions check both column presence and expected row reduction so the step cannot fail quietly again.

References
Strack et al. (2014), Impact of HbA1c Measurement on Hospital Readmission Rates, BioMed Research International.
Saito and Rehmsmeier (2015), The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets, PLOS ONE.
Detrano et al. (1989), International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease, American Journal of Cardiology.
Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python, JMLR.
