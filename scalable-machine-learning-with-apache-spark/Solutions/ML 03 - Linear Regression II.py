# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-60a5d18a-6438-4ee3-9097-5145dc31d938
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Linear Regression: Improving the Model
# MAGIC
# MAGIC In this notebook we will be adding additional features to our model, as well as discuss how to handle categorical features.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Encode categorical variables using One-Hot-Encoder method
# MAGIC * Create a Spark ML Pipeline to fit a model
# MAGIC * Evaluate a model’s performance
# MAGIC * Save and load a model using Spark ML Pipeline

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
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-b44be11f-203c-4ea4-bc3e-20e696cabb0e
# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

# COMMAND ----------

# DBTITLE 0,--i18n-f8b3c675-f8ce-4339-865e-9c64f05291a6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC Let's use the same 80/20 split with the same seed as the previous notebook so we can compare our results apples to apples (unless you changed the cluster config!)

# COMMAND ----------

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-09003d63-70c1-4fb7-a4b7-306101a88ae3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Categorical Variables
# MAGIC
# MAGIC There are a few ways to handle categorical features:
# MAGIC * Assign them a numeric value
# MAGIC * Create "dummy" variables (also known as One Hot Encoding)
# MAGIC * Generate embeddings (mainly used for textual data)
# MAGIC
# MAGIC ### One Hot Encoder
# MAGIC Here, we are going to One Hot Encode (OHE) our categorical variables. Spark doesn't have a **`dummies`** function, and OHE is a two-step process. First, we need to use <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer" target="_blank">StringIndexer</a> to map a string column of labels to an ML column of label indices.
# MAGIC
# MAGIC Then, we can apply the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html?highlight=onehotencoder#pyspark.ml.feature.OneHotEncoder" target="_blank">OneHotEncoder</a> to the output of the StringIndexer.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)

# COMMAND ----------

# DBTITLE 0,--i18n-dedd7980-1c27-4f35-9d94-b0f1a1f92839
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC Now we can combine our OHE categorical features with our numeric features.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = ohe_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-fb06fb9b-5dac-46df-aff3-ddee6dc88125
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC Now that we have all of our features, let's build a linear regression model.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="price", featuresCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-a7aabdd1-b384-45fc-bff2-f385cc7fe4ac
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC Let's put all these stages in a Pipeline. A <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html?highlight=pipeline#pyspark.ml.Pipeline" target="_blank">Pipeline</a> is a way of organizing all of our transformers and estimators.
# MAGIC
# MAGIC This way, we don't have to worry about remembering the same ordering of transformations to apply to our test dataset.

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [string_indexer, ohe_encoder, vec_assembler, lr]
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-c7420125-24be-464f-b609-1bb4e765d4ff
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Saving Models
# MAGIC
# MAGIC We can save our models to persistent storage (e.g. DBFS) in case our cluster goes down so we don't have to recompute our results.

# COMMAND ----------

pipeline_model.write().overwrite().save(DA.paths.working_dir)

# COMMAND ----------

# DBTITLE 0,--i18n-15f4623d-d99a-42d6-bee8-d7c4f79fdecb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Loading models
# MAGIC
# MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
# MAGIC
# MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic `PipelineModel` back in.

# COMMAND ----------

from pyspark.ml import PipelineModel

saved_pipeline_model = PipelineModel.load(DA.paths.working_dir)

# COMMAND ----------

# DBTITLE 0,--i18n-1303ef7d-1a57-4573-8afe-561f7730eb33
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply the Model to Test Set

# COMMAND ----------

pred_df = saved_pipeline_model.transform(test_df)

display(pred_df.select("features", "price", "prediction"))

# COMMAND ----------

# DBTITLE 0,--i18n-9497f680-1c61-4bf1-8ab4-e36af502268d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate the Model
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) How is our R2 doing?

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# DBTITLE 0,--i18n-cc0618e0-59d9-4a6d-bb90-a7945da1457e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC As you can see, our RMSE decreased when compared to the model without one-hot encoding that we trained in the previous notebook, and the R2 increased as well!

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
