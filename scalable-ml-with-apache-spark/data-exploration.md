# Importance of Data Visualisation
- Just because multiple datasets have the same mean, standard deviation and regression line, does not mean they are the same.
- They can be qualitatively different
- This is known as Anscombe's Quartet, four different data sets that have nearly identical descriptive statistics, yet have very different distributions and appear very different when graphed.
# How do we build and evaluate models?
1. Split full dataset into a training and test set
2. Each have X (features) and y (labels)
3. Model is trained off of the features
4. Test set is used to create a new entity, the predictions, which can be used to evaluate accuracy
# Important Points in Data Exploration
- **Outliers in the data** - Handling outliers can be unique to projects, but good to be informed about them.
- **Homogeneity in variance** - The variance of the feature variables needs to be similar
- **Normally distributed data** - Various statistical techniques assume normality, such as linear regressions and t-tests.
- **Zeros (missing value) in the data** - Makes the analysis more complicated. Can lead to incorrect labels
- **Collinearity in covariates** - If ignored, can lead to confusing statistical analysis.
- **Interaction between variables** - Relationship of variables will chance according to the value of other variables.
- **Independence in the dataset** - Data points in the dataset should be drawn independently. This can be analysed by modelling any spatial or temporal relationships.