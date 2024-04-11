# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-bfaf8cdf-ed63-45f7-a559-954d478bb0f7
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # De-Duping Data
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Instructions
# MAGIC
# MAGIC In this exercise, we're doing ETL on a file we've received from some customer. That file contains data about people, including:
# MAGIC
# MAGIC * first, middle and last names
# MAGIC * gender
# MAGIC * birth date
# MAGIC * Social Security number
# MAGIC * salary
# MAGIC
# MAGIC But, as is unfortunately common in data we get from this customer, the file contains some duplicate records. Worse:
# MAGIC
# MAGIC * In some of the records, the names are mixed case (e.g., "Carol"), while in others, they are uppercase (e.g., "CAROL"). 
# MAGIC * The Social Security numbers aren't consistent, either. Some of them are hyphenated (e.g., "992-83-4829"), while others are missing hyphens ("992834829").
# MAGIC
# MAGIC The name fields are guaranteed to match, if you disregard character case, and the birth dates will also match. (The salaries will match, as well,
# MAGIC and the Social Security Numbers *would* match, if they were somehow put in the same format).
# MAGIC
# MAGIC Your job is to remove the duplicate records. The specific requirements of your job are:
# MAGIC
# MAGIC * Remove duplicates. It doesn't matter which record you keep; it only matters that you keep one of them.
# MAGIC * Preserve the data format of the columns. For example, if you write the first name column in all lower-case, you haven't met this requirement.
# MAGIC * Write the result as a Parquet file, as designated by *dest_file*.
# MAGIC * The final Parquet "file" must contain 8 part files (8 files ending in ".parquet").
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** The initial dataset contains 103,000 records.<br/>
# MAGIC The de-duplicated result haves 100,000 records.

# COMMAND ----------

# DBTITLE 0,--i18n-8f030328-5dad-4ae9-bf82-30e96ffc38a5
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Getting Started
# MAGIC
# MAGIC Run the following cell to configure our "classroom."

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-c70ecf76-ddd9-4f68-8c2c-41bbf258419c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Hints
# MAGIC
# MAGIC * Use the <a href="https://spark.apache.org/docs/latest/api/python/index.html" target="_blank">API docs</a>. Specifically, you might find 
# MAGIC   <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html" target="_blank">DataFrame</a> and
# MAGIC   <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html" target="_blank">functions</a> to be helpful.
# MAGIC * It's helpful to look at the file first, so you can check the format. **`dbutils.fs.head()`** (or just **`%fs head`**) is a big help here.

# COMMAND ----------

# ANSWER

source_file = f"{DA.paths.datasets}/dataframes/people-with-dups.txt"
dest_file = f"{DA.paths.working_dir}/people.parquet"

# In case it already exists
dbutils.fs.rm(dest_file, True)

# First, let's see what the file looks like.
print(dbutils.fs.head(source_file))

# COMMAND ----------

# ANSWER

# dropDuplicates() will likely introduce a shuffle, so it helps to reduce the number of post-shuffle partitions.
spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# ANSWER

df = (spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("sep", ":")
      .csv(source_file))

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import  col, lower, translate

deduped_df = (df
              .select(col("*"),
                      lower(col("firstName")).alias("lcFirstName"),
                      lower(col("lastName")).alias("lcLastName"),
                      lower(col("middleName")).alias("lcMiddleName"),
                      translate(col("ssn"), "-", "").alias("ssnNums")
                      # regexp_replace(col("ssn"), "-", "").alias("ssnNums")
                      # regexp_replace(col("ssn"), """^(\d{3})(\d{2})(\d{4})$""", "$1-$2-$3").alias("ssnNums")
                     )
             .dropDuplicates(["lcFirstName", "lcMiddleName", "lcLastName", "ssnNums", "gender", "birthDate", "salary"])
             .drop("lcFirstName", "lcMiddleName", "lcLastName", "ssnNums")
            )

# COMMAND ----------

# ANSWER

# Now we can save the results. We'll also re-read them and count them, just as a final check.
# Just for fun, we'll use the Snappy compression codec. It's not as compact as Gzip, but it's much faster.
deduped_df.write.mode("overwrite").parquet(dest_file)

deduped_df = spark.read.parquet(dest_file)
print(f"Total Records: {deduped_df.count()}")

# COMMAND ----------

# ANSWER

display(dbutils.fs.ls(dest_file))

# COMMAND ----------

# DBTITLE 0,--i18n-3616cc9c-788f-431a-a895-fde4e0983366
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Validate Your Answer
# MAGIC
# MAGIC At the bare minimum, we can verify that you wrote the parquet file out to **dest_file** and that you have the right number of records.
# MAGIC
# MAGIC Running the following cell to confirm your result:

# COMMAND ----------

part_files = len(list(filter(lambda f: f.path.endswith(".parquet"), dbutils.fs.ls(dest_file))))

final_df = spark.read.parquet(dest_file)
final_count = final_df.count()

clearYourResults()
validateYourAnswer("01 Parquet File Exists", 1276280174, part_files)
validateYourAnswer("02 Expected 100000 Records", 972882115, final_count)
summarizeYourResults()

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
