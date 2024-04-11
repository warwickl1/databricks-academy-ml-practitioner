# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-2ab084da-06ed-457d-834a-1d19353e5c59
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Random Forests and Hyperparameter Tuning
# MAGIC
# MAGIC Now let's take a look at how to tune random forests using grid search and cross validation in order to find the optimal hyperparameters.  Using the Databricks Runtime for ML, MLflow automatically logs metrics of all your experiments with the SparkML cross-validator as well as the best fit model!
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Lesson Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC * Tune hyperparameters using Spark ML’s grid search feature
# MAGIC * Explain cross validation concepts and how to use cross validation in Spark ML pipelines
# MAGIC * Optimize a Spark ML pipeline
# MAGIC

# COMMAND ----------

# DBTITLE 0,--i18n-1e2c921e-1125-4df3-b914-d74bf7a73ab5
# MAGIC %md 
# MAGIC ## 📌 Requirements
# MAGIC
# MAGIC **Required Databricks Runtime Version:** 
# MAGIC * Please note that in order to run this notebook, you must use one of the following Databricks Runtime(s): **12.2.x-cpu-ml-scala2.12**

# COMMAND ----------

# DBTITLE 0,--i18n-6a1bb996-7b50-4f03-9bcd-3d3ec3069a6d
# MAGIC %md 
# MAGIC ## Lesson Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-67393595-40fc-4274-b9ed-40f8ef4f7db1
# MAGIC %md 
# MAGIC
# MAGIC ## Build a Model Pipeline

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40)
stages = [string_indexer, vec_assembler, rf]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

# DBTITLE 0,--i18n-4561938e-90b5-413c-9e25-ef15ba40e99c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## ParamGrid
# MAGIC
# MAGIC First let's take a look at the various hyperparameters we could tune for random forest.
# MAGIC
# MAGIC **Pop quiz:** what's the difference between a parameter and a hyperparameter?

# COMMAND ----------

print(rf.explainParams())

# COMMAND ----------

# DBTITLE 0,--i18n-819de6f9-75d2-45df-beb1-6b59ecd2cfd2
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
# MAGIC
# MAGIC Instead of a manual (ad-hoc) approach, let's use Spark's <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html?highlight=paramgridbuilder#pyspark.ml.tuning.ParamGridBuilder" target="_blank">ParamGridBuilder</a> to find the optimal hyperparameters in a more systematic approach.
# MAGIC
# MAGIC Let's define a grid of hyperparameters to test:
# MAGIC   - **`maxDepth`**: max depth of each decision tree (Use the values **`2, 5`**)
# MAGIC   - **`numTrees`**: number of decision trees to train (Use the values **`5, 10`**)
# MAGIC
# MAGIC **`addGrid()`** accepts the name of the parameter (e.g. **`rf.maxDepth`**), and a list of the possible values (e.g. **`[2, 5]`**).

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
              .build())

# COMMAND ----------

# DBTITLE 0,--i18n-9f043287-11b8-482d-8501-2f7d8b1458ea
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Cross Validation
# MAGIC
# MAGIC We are also going to use 3-fold cross validation to identify the optimal hyperparameters.
# MAGIC
# MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
# MAGIC
# MAGIC With 3-fold cross-validation, we train on 2/3 of the data, and evaluate with the remaining (held-out) 1/3. We repeat this process 3 times, so each fold gets the chance to act as the validation set. We then average the results of the three rounds.

# COMMAND ----------

# DBTITLE 0,--i18n-ec0440ab-071d-4201-be86-5eeedaf80a4f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC We pass in the **`estimator`** (pipeline), **`evaluator`**, and **`estimatorParamMaps`** to <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">CrossValidator</a> so that it knows:
# MAGIC - Which model to use
# MAGIC - How to evaluate the model
# MAGIC - What hyperparameters to set for the model
# MAGIC
# MAGIC We can also set the number of folds we want to split our data into (3), as well as setting a seed so we all have the same split in the data.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-673c9261-a861-4ace-b008-c04565230a8e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Question**: How many models are we training right now?

# COMMAND ----------

cv_model = cv.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-2d00b40f-c5e7-4089-890b-a50ccced34c6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Question**: Should we put the pipeline in the cross validator, or the cross validator in the pipeline?
# MAGIC
# MAGIC It depends if there are estimators or transformers in the pipeline. If you have things like StringIndexer (an estimator) in the pipeline, then you have to refit it every time if you put the entire pipeline in the cross validator.
# MAGIC
# MAGIC However, if there is any concern about data leakage from the earlier steps, the safest thing is to put the pipeline inside the CV, not the other way. CV first splits the data and then .fit() the pipeline. If it is placed at the end of the pipeline, we potentially can leak the info from hold-out set to train set.

# COMMAND ----------

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

stages_with_cv = [string_indexer, vec_assembler, cv]
pipeline = Pipeline(stages=stages_with_cv)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-42652818-f185-45d2-8e96-8204292fea5b
# MAGIC %md
# MAGIC In the current method, only **one MLflow run is logged**, whereas in the previous method, **five runs** were logged. This is because, in the first method, the pipeline was placed within the CrossValidator, which automatically logs all runs. However, in the second method, since **the pipeline only returns the best model without evaluating metrics**, only that single model is seen. Additionally, no evaluation metrics are logged. Essentially, the Pipeline logs all stages and directly returns the best model without performing model evaluations.

# COMMAND ----------

# DBTITLE 0,--i18n-dede990c-2551-4c07-8aad-d697ae827e71
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at the model with the best hyperparameter configuration

# COMMAND ----------

list(zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics))

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

rmse = evaluator.evaluate(pred_df)
r2 = evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# DBTITLE 0,--i18n-8f80daf2-8f0b-4cab-a8e6-4060c78d94b0
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Progress!  Looks like we're out-performing decision trees.

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
