# Apache Spark Background
1. Founded as a research project at UC Berkeley in 2009
2. Open-source unified data analytics engine for big data
3. Built-in APIs for SQL, Python, Scala, R and Java
# Spark Cluster
* Only 1 driver, which is were we send our code
* Optimises and distributes code to N workers
* Each worker runs a JVM process
# Spark's Structured Data APIs
* **RDD**: Resilient Distributed Dataset. JVM objects, collection of rows
* **DataFrame**: Distributed collection of row objects. Expression-based operations and UDFs
* **Dataset**: Internally rows, externally JVM objects. Type safe + fast (Scala or Java)
# Under the Catalyst Optimizer's Hood
1. Unresolved Logical Plan - syntax structure, making sure keywords are spelt correctly
2. Logical Plan + Optimised Logical Plan - This is where the Catalyst comes in, applies set rules to optimise the plan and devise multiple physical plans
3. Cost model - Picks the most appropriate physical plan
4. Whole stage code generation - Converts to low level Java byte code
# When to use Spark
- **Scaling out** - Data or model is too large to process on a single machine, commonly resulting in out-of-memory errors
- **Speeding up** - Data or model is processing slowly and could benefit from short processing times and faster results