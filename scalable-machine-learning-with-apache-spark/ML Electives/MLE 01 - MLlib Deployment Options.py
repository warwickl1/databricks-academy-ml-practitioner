# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-59431a59-5305-45dc-81c5-bc13132e61ce
# MAGIC %md 
# MAGIC
# MAGIC # MLlib Deployment Options
# MAGIC
# MAGIC Welcome to this lesson on Apache MLLib deployment options! In this lesson, we will explore the various ways to deploy machine learning models built using the Spark ML library. Building a model is only one part of the machine-learning process. To make a model useful, it needs to be deployed so that it can be integrated into real-world applications. In this lesson, we will discuss different deployment options available for Spark ML models.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC  - Apply a SparkML model on a simulated stream of data

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

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-58b43b1c-4722-425d-a694-41903db7b6d0
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Introduction to Deployment Types
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/301/deployment_options_mllib.png)
# MAGIC
# MAGIC There are four main deployment options:
# MAGIC * Batch pre-compute
# MAGIC * Structured streaming
# MAGIC * Low-latency model serving
# MAGIC * Mobile/embedded (outside scope of class)
# MAGIC
# MAGIC We have already seen how to do batch predictions using Spark. Now let's look at how to make predictions on streaming data.

# COMMAND ----------

# DBTITLE 0,--i18n-46846e08-4b50-4297-a871-98beaf65c3f7
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Load in Data & Model
# MAGIC
# MAGIC We are loading in a repartitioned version of our dataset (100 partitions instead of 4) to see more incremental progress of the streaming predictions.

# COMMAND ----------

from pyspark.ml.pipeline import PipelineModel

pipeline_path = f"{DA.paths.datasets}/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model"
pipeline_model = PipelineModel.load(pipeline_path)

repartitioned_path =  f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/"
schema = spark.read.parquet(repartitioned_path).schema

# COMMAND ----------

# DBTITLE 0,--i18n-6d5976b8-54b3-4379-9240-2fb9b7941f4c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Simulate streaming data
# MAGIC
# MAGIC **NOTE**: You must specify a schema when creating a streaming source DataFrame.

# COMMAND ----------

streaming_data = (spark
                 .readStream
                 .schema(schema) # Can set the schema this way
                 .option("maxFilesPerTrigger", 1)
                 .parquet(repartitioned_path))

# COMMAND ----------

# DBTITLE 0,--i18n-29c9d057-1b46-41ff-a7a0-2d80a113e7a3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Make Predictions

# COMMAND ----------

stream_pred = pipeline_model.transform(streaming_data)

# COMMAND ----------

# DBTITLE 0,--i18n-d0c54563-04fc-48f3-b739-9acc85723d51
# MAGIC %md 
# MAGIC
# MAGIC ### Save Results
# MAGIC
# MAGIC Let's save our results.

# COMMAND ----------

import re

checkpoint_dir = f"{DA.paths.working_dir}/stream_checkpoint"
# Clear out the checkpointing directory
dbutils.fs.rm(checkpoint_dir, True) 

query = (stream_pred.writeStream
                    .format("memory")
                    .option("checkpointLocation", checkpoint_dir)
                    .outputMode("append")
                    .queryName("pred_stream")
                    .start())

# COMMAND ----------

DA.block_until_stream_is_ready(query)

# COMMAND ----------

# DBTITLE 0,--i18n-3654909a-da6d-4e8e-919a-9802e8292e77
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC While this is running, take a look at the new Structured Streaming tab in the Spark UI.

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from pred_stream

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from pred_stream

# COMMAND ----------

# DBTITLE 0,--i18n-fb17c70a-c926-446c-a94a-900afc08efff
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now that we are done, make sure to stop the stream

# COMMAND ----------

for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop()             # Stop the active stream
    stream.awaitTermination() # Wait for it to actually stop


# COMMAND ----------

# DBTITLE 0,--i18n-622245a0-07c0-43ee-967c-41cb4a601152
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## What about Model Export?
# MAGIC
# MAGIC * <a href="https://onnx.ai/" target="_blank">ONNX</a>
# MAGIC   * ONNX is very popular in the deep learning community allowing developers to switch between libraries and languages, but only has experimental support for MLlib.
# MAGIC * DIY (Reimplement it yourself)
# MAGIC   * Error-prone, fragile
# MAGIC * 3rd party libraries
# MAGIC   * See XGBoost notebook
# MAGIC   * <a href="https://www.h2o.ai/products/h2o-sparkling-water/" target="_blank">H2O</a>

# COMMAND ----------

# DBTITLE 0,--i18n-39b0e95b-29e0-462f-a7ec-17bb6c5469ef
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Low-Latency Serving Solutions
# MAGIC
# MAGIC Low-latency serving can operate as quickly as tens to hundreds of milliseconds.  Custom solutions are normally backed by Docker and/or Flask (though Flask generally isn't recommended in production unless significant precautions are taken).  Managed solutions also include:<br><br>
# MAGIC
# MAGIC * <a href="https://docs.databricks.com/applications/mlflow/model-serving.html" target="_blank">MLflow Model Serving</a>
# MAGIC * <a href="https://azure.microsoft.com/en-us/services/machine-learning/" target="_blank">Azure Machine Learning</a>
# MAGIC * <a href="https://aws.amazon.com/sagemaker/" target="_blank">SageMaker</a>

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
