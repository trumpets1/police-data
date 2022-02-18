# Police Data

Download CSV from [this Police Data](https://data.sanjoseca.gov/dataset/police-calls-for-service).

This is just some data analysis and machine learning that I chose to do on San Jose's police data.

For machine learning, I'm just comparing the performance of the naive Bayes classifier, the one-class SVM (with stochastic gradient descent), and gradient boosting.

Utilizes scikit-learn, pandas, numpy, and matplotlib.

## CSV generator notebook

These notebooks contain data processing/generation steps for each component of this whole project.

## Exploration notebooks

These notebooks just contain simple statistics and visualizations for the data, e.g. frequency of crimes plotted over the months.

## Classifier Comparison

Where I really get into prediction systems. Again -- uses (categorical) naive Bayes classifier, one-class SVM, and gradient boosting.

Models found in `models/predict.py`.
