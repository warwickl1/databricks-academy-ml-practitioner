# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-0f0f211a-70ba-4432-ab87-19bf7c8fc6cc
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # AutoML Lab
# MAGIC
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/automl.html" target="_blank">Databricks AutoML</a> helps you automatically build machine learning models both through a UI and programmatically. It prepares the dataset for model training and then performs and records a set of trials (using HyperOpt), creating, tuning, and evaluating multiple models. 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lab, you should be able to;
# MAGIC
# MAGIC * Utilize AutoML to automatically train and tune machine learning models
# MAGIC * Create and run a model using AutoML UI
# MAGIC * Interpret the results of an AutoML run and select the best performing model

# COMMAND ----------

# DBTITLE 0,--i18n-9debeda3-f818-4529-b3e5-54ca09d46a76
# MAGIC %md 
# MAGIC ## Lab Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-9178a13f-aeaf-49cb-ac27-600e7cea0077
# MAGIC %md 
# MAGIC
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Currently, AutoML uses a combination of XGBoost and sklearn (only single node models) but optimizes the hyperparameters within each.

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-af913436-4be4-4a26-8381-d40d4e1af9d2
# MAGIC %md 
# MAGIC
# MAGIC Instead of programmatically building our models, we can also use the UI. But first we need to register our dataset as a table.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.schema_name}")
train_df.write.mode("overwrite").saveAsTable(f"{DA.schema_name}.autoMLTable")

# COMMAND ----------

# DBTITLE 0,--i18n-2f854d06-800c-428c-8add-aece6c9a91b6
# MAGIC %md 
# MAGIC
# MAGIC ## Create an AutoML Experiment 
# MAGIC
# MAGIC * Navigate to the sidebar on the workspace homepage.
# MAGIC * Click on the **Experiments** tab
# MAGIC * Click on **Create AutoML Experiment** button on the top-right corner
# MAGIC * Configure the AutoML experiment

# COMMAND ----------

# DBTITLE 0,--i18n-98f64ede-5b15-442b-8346-874e0fdea6b5
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Select **`regression`** as the problem type, as well as the table we created in the previous cell. Then, select **`price`** as the column to predict.
# MAGIC
# MAGIC <img src="http://files.training.databricks.com/images/Scalable-ML-AutoML-UI-v2.png" alt="ui" width="750"/>

# COMMAND ----------

# DBTITLE 0,--i18n-4e561687-2509-4084-bd33-4221cb047eba
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC In the advanced configuration dropdown, change the evaluation metric to rmse, timeout to 5 minutes, and the maximum number of runs to 20.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/Scalable-ML-AutoML-Advanced-v2.png" alt="advanced" width="500"/>

# COMMAND ----------

# DBTITLE 0,--i18n-b15305f8-04cd-422f-a1da-ad7640b3846b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, we can start our run. Once completed, we can view the tuned model by clicking on the edit best model button.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/Scalable-ML-AutoML-Results-Best-Model-v2.png" alt="results" width="1000"/>

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
