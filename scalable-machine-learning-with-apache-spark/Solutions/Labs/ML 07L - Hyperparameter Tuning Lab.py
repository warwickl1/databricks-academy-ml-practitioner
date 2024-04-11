# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-b778c8d0-84e6-4192-a921-b9b60fd20d9b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Hyperparameter Tuning with Random Forests
# MAGIC
# MAGIC In this lab, you will convert the Airbnb problem to a classification dataset, build a random forest classifier, and tune some hyperparameters of the random forest.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lab, you should be able to;
# MAGIC
# MAGIC * Perform grid search on a random forest based model
# MAGIC * Generate feature importance scores and classification metrics for a random forest model
# MAGIC * Identify differences between scikit-learn's and Spark ML's Random Forest implementation
# MAGIC
# MAGIC
# MAGIC  
# MAGIC You can read more about the distributed implementation of Random Forests in the Spark <a href="https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/tree/impl/RandomForest.scala#L42" target="_blank">source code</a>.

# COMMAND ----------

# DBTITLE 0,--i18n-40dbd041-1a78-4f06-b83f-e9ad3d51d6ed
# MAGIC %md 
# MAGIC ## Lab Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-02dc0920-88e1-4f5b-886c-62b8cc02d1bb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## From Regression to Classification
# MAGIC
# MAGIC In this case, we'll turn the Airbnb housing dataset into a classification problem to **classify between high and low price listings.**  Our **`class`** column will be:<br><br>
# MAGIC
# MAGIC - **`0`** for a low cost listing of under $150
# MAGIC - **`1`** for a high cost listing of $150 or more

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"

airbnb_df = (spark
            .read
            .format("delta")
            .load(file_path)
            .withColumn("priceClass", (col("price") >= 150).cast("int"))
            .drop("price")
           )

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "priceClass"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-e3bb8033-43ea-439c-a134-36bedbeff408
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Why can't we OHE?
# MAGIC
# MAGIC **Question:** What would go wrong if we One Hot Encoded our variables before passing them into the random forest?
# MAGIC
# MAGIC **HINT:** Think about what would happen to the "randomness" of feature selection.

# COMMAND ----------

# DBTITLE 0,--i18n-0e9bdc2f-0d8d-41cb-9509-47833d66bc5e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Random Forest
# MAGIC
# MAGIC Create a Random Forest classifer called **`rf`** with the **`labelCol=priceClass`**, **`maxBins=40`**, and **`seed=42`** (for reproducibility).
# MAGIC
# MAGIC It's under **`pyspark.ml.classification.RandomForestClassifier`** in Python.

# COMMAND ----------

# ANSWER
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="priceClass", maxBins=40, seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-7f3962e7-51b8-4477-9599-2465ab94a049
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Grid Search
# MAGIC
# MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
# MAGIC
# MAGIC Let's use Spark's <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder" target="_blank">ParamGridBuilder</a> to find the optimal hyperparameters in a more systematic approach. Call this variable **`param_grid`**.
# MAGIC
# MAGIC Let's define a grid of hyperparameters to test:
# MAGIC   - maxDepth: max depth of the decision tree (Use the values **`2, 5, 10`**)
# MAGIC   - numTrees: number of decision trees (Use the values **`10, 20, 100`**)
# MAGIC
# MAGIC **`addGrid()`** accepts the name of the parameter (e.g. **`rf.maxDepth`**), and a list of the possible values (e.g. **`[2, 5, 10]`**).

# COMMAND ----------

# ANSWER

from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5, 10])
              .addGrid(rf.numTrees, [10, 20, 100])
              .build())

# COMMAND ----------

# DBTITLE 0,--i18n-e1862bae-e31e-4f5a-ab0e-926261c4e27b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluator
# MAGIC
# MAGIC In the past, we used a **`RegressionEvaluator`**.  For classification, we can use a <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html?highlight=binaryclass#pyspark.ml.evaluation.BinaryClassificationEvaluator" target="_blank">BinaryClassificationEvaluator</a> if we have two classes or <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html?highlight=multiclass#pyspark.ml.evaluation.MulticlassClassificationEvaluator" target="_blank">MulticlassClassificationEvaluator</a> for more than two classes.
# MAGIC
# MAGIC Create a **`BinaryClassificationEvaluator`** with **`areaUnderROC`** as the metric.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic" target="_blank">Read more on ROC curves here.</a>  In essence, it compares true positive and false positives.

# COMMAND ----------

# ANSWER
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="priceClass", metricName="areaUnderROC")

# COMMAND ----------

# DBTITLE 0,--i18n-ea1c0e11-125d-4067-bd70-0bd6c7ca3cdb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cross Validation
# MAGIC
# MAGIC We are going to do 3-Fold cross-validation and set the **`seed`**=42 on the cross-validator for reproducibility.
# MAGIC
# MAGIC Put the Random Forest in the CV to speed up the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">cross validation</a> (as opposed to the pipeline in the CV).

# COMMAND ----------

# ANSWER
from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid,
                    numFolds=3, seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-1f8cebd5-673c-4513-b73b-b64b0a56297c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC Let's fit the pipeline with our cross validator to our training data (this may take a few minutes).

# COMMAND ----------

stages = [string_indexer, vec_assembler, cv]

pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-70cdbfa3-0dd7-4f23-b755-afc0dadd7eb2
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Hyperparameter
# MAGIC
# MAGIC Which hyperparameter combination performed the best?

# COMMAND ----------

cv_model = pipeline_model.stages[-1]
rf_model = cv_model.bestModel

# list(zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics))

print(rf_model.explainParams())

# COMMAND ----------

# DBTITLE 0,--i18n-11e6c47a-ddb1-416d-92a5-2f61340f9a5e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Feature Importance

# COMMAND ----------

import pandas as pd

pandas_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), rf_model.featureImportances)), columns=["feature", "importance"])
top_features = pandas_df.sort_values(["importance"], ascending=False)
top_features

# COMMAND ----------

# DBTITLE 0,--i18n-ae7e312e-d32b-4b02-97ff-ad4d2c737892
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Do those features make sense? Would you use those features when picking an Airbnb rental?

# COMMAND ----------

# DBTITLE 0,--i18n-950eb40f-b1d2-4e7f-8b07-76faff6b8186
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply Model to Test Set

# COMMAND ----------

# ANSWER

pred_df = pipeline_model.transform(test_df)
area_under_roc = evaluator.evaluate(pred_df)
print(f"Area under ROC is {area_under_roc:.2f}")

# COMMAND ----------

# DBTITLE 0,--i18n-01974668-f242-4b8a-ac80-adda3b98392d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Save Model
# MAGIC
# MAGIC Save the model to **`DA.paths.working_dir`** (variable defined in Classroom-Setup)

# COMMAND ----------

# ANSWER
pipeline_model.write().overwrite().save(DA.paths.working_dir)

# COMMAND ----------

# DBTITLE 0,--i18n-f5fdf1a9-2a65-4252-aa76-18807dbb3a9d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Sklearn vs SparkML
# MAGIC
# MAGIC <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">Sklearn RandomForestRegressor</a> vs <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html?highlight=randomfore#pyspark.ml.regression.RandomForestRegressor" target="_blank">SparkML RandomForestRegressor</a>.
# MAGIC
# MAGIC Look at these params in particular:
# MAGIC * **n_estimators** (sklearn) vs **numTrees** (SparkML)
# MAGIC * **max_depth** (sklearn) vs **maxDepth** (SparkML)
# MAGIC * **max_features** (sklearn) vs **featureSubsetStrategy** (SparkML)
# MAGIC * **maxBins** (SparkML only)
# MAGIC
# MAGIC What do you notice that is different?

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md 
# MAGIC
# MAGIC ## Classroom Cleanup
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
