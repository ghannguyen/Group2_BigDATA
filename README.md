<div align="center">

# Walmart Big Data Analytics

### A Big Data project using Hadoop, HDFS, Apache Spark, Spark SQL, and Spark MLlib

This project analyzes Walmart Store Sales data through a complete Big Data pipeline, including raw data storage on HDFS, Spark-based preprocessing, analytical queries using Spark SQL, machine learning with Spark MLlib, and performance optimization experiments.

<br>

![Hadoop](https://img.shields.io/badge/Hadoop-3.x-FFCC00?style=for-the-badge\&logo=apachehadoop\&logoColor=black)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.x-E25A1C?style=for-the-badge\&logo=apachespark\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-SQL%20%7C%20MLlib-FDEE21?style=for-the-badge\&logo=apachespark\&logoColor=black)
![GitHub](https://img.shields.io/badge/GitHub-Team%20Collaboration-181717?style=for-the-badge\&logo=github\&logoColor=white)

<br>

[Source Code](https://github.com/ghannguyen/Group2_BigDATA)

</div>

---

## Project Information

| Category          | Details                                                    |
| ----------------- | ---------------------------------------------------------- |
| Project Name      | Walmart Big Data Analytics                                 |
| Course            | Big Data                                                   |
| University        | University of Economics Ho Chi Minh City                   |
| School            | School of Technology and Design                            |
| Major             | Business Information Systems                               |
| Team              | Group 2                                                    |
| Dataset           | Walmart Store Sales                                        |
| Main Technologies | Hadoop, HDFS, Apache Spark, Spark SQL, Spark MLlib         |
| Status            | Completed source code, screenshots, and benchmark evidence |

---

## Table of Contents

* [Project Overview](#project-overview)
* [Business Understanding](#business-understanding)
* [Dataset](#dataset)
* [Data Pipeline](#data-pipeline)
* [System Architecture](#system-architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [HDFS Storage Design](#hdfs-storage-design)
* [Installation and Environment Setup](#installation-and-environment-setup)
* [How to Run the Project](#how-to-run-the-project)
* [Spark SQL Analysis](#spark-sql-analysis)
* [Spark MLlib Models](#spark-mllib-models)
* [Performance Optimization](#performance-optimization)
* [Screenshots Evidence](#screenshots-evidence)
* [GitHub Collaboration](#github-collaboration)
* [Important Notes](#important-notes)
* [Team Members and Contributions](#team-members-and-contributions)
* [Project Status](#project-status)
* [Future Improvements](#future-improvements)
* [Lessons Learned](#lessons-learned)
* [Acknowledgements](#acknowledgements)
* [License](#license)

---

## Project Overview

Walmart Big Data Analytics is an academic Big Data project built to demonstrate how Hadoop, HDFS, Apache Spark, Spark SQL, and Spark MLlib can be used to process and analyze large-scale retail sales data.

The project focuses on the full data lifecycle:

* Checking dataset requirements
* Uploading raw datasets to HDFS
* Reading raw data directly from HDFS using Spark
* Joining and preprocessing multiple CSV files
* Writing processed data to HDFS in Parquet format
* Executing business-oriented Spark SQL queries
* Building machine learning models with Spark MLlib
* Performing cache and repartition benchmark experiments
* Saving models and benchmark outputs back to HDFS
* Capturing evidence screenshots for report and presentation

The final processed dataset contains **421,570 rows** and **20 columns**, satisfying the course requirement of more than 100,000 records and more than 10 attributes.

---

## Business Understanding

Retail businesses generate large volumes of transactional, store, and external condition data. Analyzing this data helps businesses understand sales performance, seasonal patterns, holiday effects, store differences, and factors that influence revenue.

This project uses Walmart sales data to answer business questions such as:

1. Which stores and departments generate the highest sales?
2. How do monthly and yearly sales patterns change over time?
3. How do holidays affect weekly sales?
4. Which stores have similar sales and operational characteristics?
5. Can weekly sales be predicted using historical and external features?
6. How can Spark caching and repartitioning improve analytical performance?

The project simulates a Big Data analytics workflow for retail decision support.

---

## Dataset

The project uses the Walmart Store Sales dataset with four raw CSV files.

### Raw Files

| File           |    Rows | Columns | Description                                                                                       |
| -------------- | ------: | ------: | ------------------------------------------------------------------------------------------------- |
| `train.csv`    | 421,570 |       5 | Historical weekly sales by store and department                                                   |
| `features.csv` |   8,190 |      12 | External features such as temperature, fuel price, CPI, unemployment, markdowns, and holiday flag |
| `stores.csv`   |      45 |       3 | Store metadata including store type and size                                                      |
| `test.csv`     | 115,064 |       4 | Test structure for future sales prediction                                                        |

### Processed Dataset

The processed dataset is created by joining:

* `train.csv`
* `features.csv`
* `stores.csv`

Join keys:

```text
Store
Date
IsHoliday
```

Additional engineered columns:

| Column           | Description                                      |
| ---------------- | ------------------------------------------------ |
| `Year`           | Year extracted from Date                         |
| `Month`          | Month extracted from Date                        |
| `WeekOfYear`     | Week number extracted from Date                  |
| `markdown_total` | Total markdown value from MarkDown1 to MarkDown5 |

Final processed dataset:

| Attribute |                                               Value |
| --------- | --------------------------------------------------: |
| Rows      |                                             421,570 |
| Columns   |                                                  20 |
| Format    |                                             Parquet |
| HDFS Path | `/bigdata/walmart/processed/walmart_sales_enriched` |

---

## Data Pipeline

The data pipeline contains five main phases.

```text
Raw CSV Files
     |
     v
Upload to HDFS
     |
     v
Read Raw Data with Spark
     |
     v
Preprocess and Join Data
     |
     v
Write Processed Parquet to HDFS
     |
     v
Spark SQL + MLlib + Benchmark Experiments
```

### Pipeline Steps

| Step | Script                                    | Input                   | Output                                     |
| ---- | ----------------------------------------- | ----------------------- | ------------------------------------------ |
| 1    | `src/00_check_dataset.py`                 | Raw and processed data  | Dataset requirement verification           |
| 2    | `src/01_load_to_hdfs.sh`                  | Local raw CSV files     | Raw files on HDFS                          |
| 3    | `src/02_read_from_hdfs.py`                | Raw HDFS files          | Schema, count, and null checks             |
| 4    | `src/03_data_preprocess.py`               | Train, features, stores | Processed Parquet dataset                  |
| 5    | `src/03b_verify_processed_hdfs.py`        | Processed Parquet       | Row and column verification                |
| 6    | `src/04_spark_sql_queries.py`             | Processed Parquet       | 12 Spark SQL query results                 |
| 7    | `src/05_mllib_weekly_sales_prediction.py` | Processed Parquet       | RandomForest model and metrics             |
| 8    | `src/06_mllib_store_clustering.py`        | Processed Parquet       | KMeans clustering model and visualizations |
| 9    | `src/07_performance_benchmark.py`         | Processed Parquet       | SQL cache benchmark                        |
| 10   | `src/08_partition_benchmark.py`           | Processed Parquet       | Partition strategy benchmark               |

---

## System Architecture

The project follows a single-node pseudo-distributed Big Data architecture.

```text
Local Dataset
     |
     v
Hadoop HDFS
     |
     v
Apache Spark
     |
     +--------------------+
     |                    |
     v                    v
Spark SQL              Spark MLlib
     |                    |
     v                    v
Query Results       Models and Metrics
     |                    |
     +---------+----------+
               |
               v
Benchmark Outputs and Screenshots
```

### Component Responsibilities

| Component          | Responsibility                                                 |
| ------------------ | -------------------------------------------------------------- |
| Local File System  | Stores original dataset files before upload                    |
| Hadoop HDFS        | Stores raw data, processed data, models, and benchmark outputs |
| Apache Spark       | Reads, processes, analyzes, and writes data                    |
| Spark SQL          | Executes analytical business queries                           |
| Spark MLlib        | Builds prediction and clustering models                        |
| GitHub             | Manages source code, branches, commits, and collaboration      |
| Screenshots Folder | Stores evidence for report and presentation                    |

---

## Technology Stack

### Big Data and Processing

* Apache Hadoop
* HDFS
* Apache Spark
* Spark SQL
* Spark MLlib
* PySpark

### Programming and Analysis

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Development Tools

* Git
* GitHub
* Visual Studio Code
* Terminal / Shell Script
* macOS development environment

---

## Project Structure

```text
Group2_BigDATA/
├── data/
│   ├── raw/
│   └── processed/
├── environment/
│   ├── config_files/
│   │   ├── core-site.xml
│   │   ├── hdfs-site.xml
│   │   ├── mapred-site.xml
│   │   └── yarn-site.xml
│   ├── hadoop_installation_steps.md
│   ├── spark_installation_steps.md
│   └── hdfs_commands.md
├── screenshots/
│   ├── 00_dataset_check/
│   ├── 01_environment/
│   ├── 02_hadoop_hdfs/
│   ├── 03_spark_processing/
│   ├── 04_spark_sql/
│   ├── 05_mllib/
│   ├── 06_performance_optimization/
│   ├── 07_github_project/
│   └── 08_report_slide/
├── src/
│   ├── 00_check_dataset.py
│   ├── 01_load_to_hdfs.sh
│   ├── 02_read_from_hdfs.py
│   ├── 03_data_preprocess.py
│   ├── 03b_verify_processed_hdfs.py
│   ├── 04_spark_sql_queries.py
│   ├── 05_mllib_weekly_sales_prediction.py
│   ├── 06_mllib_store_clustering.py
│   ├── 07_performance_benchmark.py
│   └── 08_partition_benchmark.py
├── .gitignore
└── README.md
```

---

## HDFS Storage Design

| Data Type                         | HDFS Path                                            |
| --------------------------------- | ---------------------------------------------------- |
| Raw data                          | `/bigdata/walmart/raw`                               |
| Processed Parquet data            | `/bigdata/walmart/processed/walmart_sales_enriched`  |
| RandomForest model                | `/bigdata/walmart/models/rf_weekly_sales_prediction` |
| KMeans model                      | `/bigdata/walmart/models/kmeans_store_clustering`    |
| RandomForest prediction benchmark | `/bigdata/walmart/benchmarks/rf_pred_benchmark`      |
| SQL cache benchmark               | `/bigdata/walmart/benchmarks/sql_cache_benchmark`    |
| Partition benchmark               | `/bigdata/walmart/benchmarks/partition_benchmark`    |

---

## Installation and Environment Setup

### Prerequisites

Make sure the following tools are installed:

* Java JDK 17
* Apache Hadoop 3.x
* Apache Spark 4.x
* Python 3.10 or higher
* PySpark
* Git
* Visual Studio Code or another code editor

### Python Libraries

Install the required Python libraries:

```bash
python3 -m pip install numpy pandas matplotlib seaborn scikit-learn pyspark
```

On macOS with Homebrew Python 3.11, the project can use:

```bash
/opt/homebrew/bin/python3.11 -m pip install numpy pandas matplotlib seaborn scikit-learn pyspark
```

Set PySpark Python if needed:

```bash
export PYSPARK_PYTHON=/opt/homebrew/bin/python3.11
export PYSPARK_DRIVER_PYTHON=/opt/homebrew/bin/python3.11
export SPARK_LOCAL_IP=127.0.0.1
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone git@github.com:ghannguyen/Group2_BigDATA.git
cd Group2_BigDATA
```

---

### 2. Start Hadoop and YARN

```bash
start-dfs.sh
start-yarn.sh
jps
```

Expected services:

```text
NameNode
DataNode
SecondaryNameNode
ResourceManager
NodeManager
```

---

### 3. Upload Raw Data to HDFS

```bash
bash src/01_load_to_hdfs.sh
```

Check uploaded files:

```bash
hdfs dfs -ls /bigdata/walmart/raw
```

Expected files:

```text
features.csv
stores.csv
test.csv
train.csv
```

---

### 4. Read Raw Data from HDFS

```bash
spark-submit src/02_read_from_hdfs.py
```

This script verifies that Spark can read raw CSV files directly from HDFS.

---

### 5. Preprocess Data and Write Parquet

```bash
spark-submit src/03_data_preprocess.py
```

This script performs:

* Join between train and features data
* Join with stores data
* Date conversion
* Feature engineering
* Missing value handling
* Parquet writing to HDFS

---

### 6. Verify Processed Dataset

```bash
spark-submit src/03b_verify_processed_hdfs.py
```

Expected result:

```text
Rows: 421570
Columns: 20
```

---

### 7. Check Dataset Requirement

```bash
spark-submit src/00_check_dataset.py
```

The dataset must satisfy:

* At least 100,000 records
* At least 10 attributes

---

### 8. Run Spark SQL Queries

```bash
spark-submit src/04_spark_sql_queries.py
```

This script runs 12 Spark SQL queries for business analysis.

---

### 9. Run RandomForest Sales Prediction

```bash
spark-submit src/05_mllib_weekly_sales_prediction.py
```

Main outputs:

* RMSE
* MAE
* R²
* Prediction sample
* Actual vs predicted chart
* Residual distribution chart
* Model saved to HDFS
* Prediction benchmark saved to HDFS

---

### 10. Run KMeans Store Clustering

```bash
spark-submit src/06_mllib_store_clustering.py
```

Main outputs:

* Store-level feature dataset
* KMeans cluster assignment
* Silhouette score
* Cluster summary
* Cluster visualizations
* Model saved to HDFS

---

### 11. Run SQL Cache Benchmark

```bash
spark-submit src/07_performance_benchmark.py
```

Main outputs:

* No-cache runtime
* Cached runtime
* Speedup percentage
* Physical plan with `InMemoryTableScan`
* Spark UI Storage evidence
* Benchmark CSV saved to HDFS

---

### 12. Run Partition Benchmark

```bash
spark-submit src/08_partition_benchmark.py
```

Main outputs:

* Default partition benchmark
* Repartition by Store benchmark
* Warm-up and materialization time
* Physical plan comparison
* Benchmark CSV saved to HDFS
* Interpretation of repartition effect

---

## Spark SQL Analysis

The project includes 12 Spark SQL queries. These queries are designed to be business-oriented and more advanced than simple SELECT statements.

### Query Techniques

| Technique       | Usage                                               |
| --------------- | --------------------------------------------------- |
| Aggregation     | Sales summary by store, department, month, and year |
| `CASE WHEN`     | Conditional sales and holiday analysis              |
| CTE             | Cleaner multi-step query logic                      |
| Window Function | Ranking and moving average                          |
| `DENSE_RANK`    | Ranking departments or stores by sales              |
| `NTILE`         | Segmenting sales performance into groups            |
| Moving Average  | Tracking sales trends over time                     |
| `CUBE`          | Multidimensional aggregation                        |
| Physical Plan   | Inspecting Spark execution plan                     |

### Example Business Questions

* Which stores have the highest total weekly sales?
* Which departments perform best during holidays?
* How do monthly sales change across years?
* Which stores have the strongest sales ranking over time?
* How does sales performance differ by store type?
* What are the multidimensional sales patterns by year, month, type, and holiday?

---

## Spark MLlib Models

The project includes two Spark MLlib models.

---

### 1. RandomForest Weekly Sales Prediction

The first MLlib task predicts `Weekly_Sales`.

#### Model Type

```text
Supervised Learning - Regression
```

#### Pipeline Components

| Component             | Purpose                                |
| --------------------- | -------------------------------------- |
| Imputer               | Handles missing numeric values         |
| StringIndexer         | Converts categorical store type values |
| OneHotEncoder         | Encodes categorical features           |
| VectorAssembler       | Combines input features                |
| RandomForestRegressor | Predicts weekly sales                  |

#### Input Features

Examples of input features:

* Store
* Dept
* IsHoliday
* Temperature
* Fuel_Price
* MarkDown1 to MarkDown5
* CPI
* Unemployment
* Type
* Size
* Year
* Month
* WeekOfYear
* markdown_total

#### Evaluation Metrics

| Metric | Meaning                      |
| ------ | ---------------------------- |
| RMSE   | Root Mean Squared Error      |
| MAE    | Mean Absolute Error          |
| R²     | Coefficient of Determination |

#### Output

| Output               | Path                                                 |
| -------------------- | ---------------------------------------------------- |
| RandomForest model   | `/bigdata/walmart/models/rf_weekly_sales_prediction` |
| Prediction benchmark | `/bigdata/walmart/benchmarks/rf_pred_benchmark`      |
| Visualizations       | `screenshots/05_mllib/`                              |

#### Note

The benchmark in this script compares repeated prediction evaluation actions before and after caching. It does not measure RandomForest training speed.

---

### 2. KMeans Store Clustering

The second MLlib task clusters Walmart stores based on store-level characteristics.

#### Model Type

```text
Unsupervised Learning - Clustering
```

#### Important Interpretation

The original sales dataset contains 421,570 rows. For KMeans, the data is first aggregated into store-level features. The model is then trained on 45 store records.

This means the Big Data processing happens during the aggregation stage, while the final clustering model operates at store level.

#### Pipeline Components

| Component               | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| Store-level aggregation | Creates analytical features for each store |
| Imputer                 | Handles missing values                     |
| StringIndexer           | Converts store type                        |
| OneHotEncoder           | Encodes categorical features               |
| StandardScaler          | Standardizes feature values                |
| KMeans                  | Clusters stores into groups                |

#### Output

| Output                 | Path                                              |
| ---------------------- | ------------------------------------------------- |
| KMeans model           | `/bigdata/walmart/models/kmeans_store_clustering` |
| Cluster visualizations | `screenshots/05_mllib/`                           |

---

## Performance Optimization

The project includes three performance optimization experiments.

---

### 1. Spark SQL Cache Benchmark

Script:

```text
src/07_performance_benchmark.py
```

Purpose:

* Compare query runtime before and after caching.
* Demonstrate when cache/persist can improve repeated analytical workloads.

Method:

1. Load processed Parquet data.
2. Run three analytical queries without cache.
3. Cache the processed DataFrame.
4. Materialize cache using `count()`.
5. Run the same three queries again.
6. Compare runtime and speedup.
7. Save benchmark result to HDFS.

Evidence:

* Runtime comparison table
* Spark physical plan
* `InMemoryTableScan`
* Spark UI Storage tab
* HDFS benchmark CSV

---

### 2. RandomForest Prediction Cache Benchmark

Script:

```text
src/05_mllib_weekly_sales_prediction.py
```

Purpose:

* Compare repeated prediction evaluation runtime before and after caching.
* Demonstrate cache benefit when the same prediction DataFrame is reused for multiple metrics.

Metrics:

* RMSE
* MAE
* R²
* Evaluation runtime

Important note:

This benchmark focuses on evaluation runtime, not model training runtime.

---

### 3. Partition Strategy Benchmark

Script:

```text
src/08_partition_benchmark.py
```

Purpose:

* Compare default partition strategy with repartitioning by Store.
* Evaluate whether repartitioning helps Store-based analytical workloads.

Compared strategies:

| Strategy             | Description                     |
| -------------------- | ------------------------------- |
| Default              | Original DataFrame partitioning |
| Repartition by Store | `repartition(8, "Store")`       |

Workloads:

| Workload | Description                            |
| -------- | -------------------------------------- |
| Agg 1    | Sales statistics by Store              |
| Agg 2    | Monthly sales rank within each Store   |
| Agg 3    | Sales statistics by Type and IsHoliday |

Interpretation:

Repartitioning by Store can improve Store-based workloads because related records are grouped by the same partitioning key. However, repartitioning introduces a shuffle cost. Therefore, repartitioning is most useful when the repartitioned DataFrame is reused enough times to offset the initial materialization cost.

---

## Screenshots Evidence

The `screenshots/` directory stores evidence for the report and presentation.

| Folder                        | Description                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `00_dataset_check`            | Dataset requirement verification                                                   |
| `01_environment`              | Hadoop, HDFS, YARN, and Spark environment                                          |
| `02_hadoop_hdfs`              | Raw and processed data on HDFS                                                     |
| `03_spark_processing`         | Spark read, preprocessing, and verification                                        |
| `04_spark_sql`                | Spark SQL query results                                                            |
| `05_mllib`                    | RandomForest and KMeans results                                                    |
| `06_performance_optimization` | Cache benchmark, Spark UI Storage, partition benchmark, and HDFS benchmark outputs |
| `07_github_project`           | GitHub contributors, commits, branches, and project collaboration                  |
| `08_report_slide`             | Report and presentation evidence                                                   |

---

## GitHub Collaboration

The team uses GitHub for source code management and collaboration.

### Collaboration Evidence

| Evidence             | Description                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Branches             | Separate work areas for dataset check, data engineering, Spark SQL, MLlib, performance benchmark, and screenshots |
| Commits              | Track implementation history and code changes                                                                     |
| Pull Requests        | Merge completed features into the develop branch                                                                  |
| Contributors Insight | Shows member contribution and commit activity                                                                     |
| Pulse Insight        | Shows repository activity, pull requests, commits, and changed files                                              |

### Branch Examples

```text
develop
feature/data-engineering
feature/performance-benchmark
Han_sql_queries
Han_mllib_model
check_dataset
screenshots_structure
```

---

## Important Notes

### Execution Mode

Hadoop, HDFS, and YARN are configured in pseudo-distributed mode.

Most Spark scripts depend on the `spark-submit` configuration used during execution. However, the following benchmark scripts explicitly use Spark local mode:

```text
src/07_performance_benchmark.py
src/08_partition_benchmark.py
```

Therefore, the benchmark results reflect single-node local-mode performance, not a full multi-node YARN cluster benchmark.

---

### Missing Values

During preprocessing, some numeric missing values are handled before writing the processed Parquet dataset. For business interpretation, missing-value handling should be explained carefully in the report.

---

### KMeans Data Level

KMeans is trained on store-level aggregated data, not directly on all 421,570 transaction-level records.

---

### Repartition Benchmark Interpretation

The aggregation runtime and repartition materialization time should be interpreted separately. Repartitioning is not always faster for every workload. It is more beneficial when the partition key matches the query workload and when the repartitioned DataFrame is reused across multiple actions.

---

## Team Members and Contributions

| Member          | Main Contributions                                                                                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ghannguyen      | Hadoop/HDFS setup, data engineering pipeline, preprocessing, processed Parquet verification, MLlib implementation, RandomForest evaluation, screenshot evidence, documentation support |
| tntuyen183      | Spark SQL queries, analytical query development, performance benchmark implementation, cache/persist experiment, partition benchmark, GitHub collaboration                             |
| nghia122005-cmy | Dataset checking, requirement verification, supporting scripts, testing outputs, GitHub collaboration, report evidence support                                                         |

> The contribution table can be adjusted based on the final team agreement and GitHub commit history.

---

## Project Status

```text
Status: Completed
Version: 1.0
Completed: June 2026
```

Completed components:

* Hadoop and HDFS setup
* Raw data upload to HDFS
* Spark read from HDFS
* Data preprocessing and Parquet writing
* Dataset requirement verification
* 12 Spark SQL analytical queries
* RandomForest weekly sales prediction
* KMeans store clustering
* SQL cache benchmark
* RandomForest prediction cache benchmark
* Partition strategy benchmark
* HDFS benchmark outputs
* Evidence screenshots
* GitHub collaboration evidence

---

## Future Improvements

* Run Spark jobs on a real multi-node cluster
* Run performance benchmarks multiple times and report average, minimum, and maximum runtime
* Add AQE on/off benchmark comparison
* Add partitioned write by Year or Store
* Tune executor memory, executor cores, and shuffle partitions
* Improve missing-value treatment using median or domain-based imputation
* Use time-based train/test split for sales forecasting
* Add more advanced ML models such as Gradient-Boosted Trees
* Add dashboard visualization for business users
* Add automated test scripts and CI workflow

---

## Lessons Learned

Through this project, the team gained practical experience in:

* Setting up Hadoop, HDFS, and YARN
* Uploading and managing files on HDFS
* Reading and writing distributed data with Spark
* Preprocessing and joining multiple datasets
* Using Parquet for efficient analytical storage
* Writing business-oriented Spark SQL queries
* Applying Spark SQL window functions, ranking, CTE, and multidimensional analysis
* Building supervised and unsupervised MLlib models
* Evaluating machine learning models using RMSE, MAE, R², and Silhouette score
* Using cache/persist and interpreting Spark physical plans
* Comparing partition strategies and understanding shuffle cost
* Saving models and benchmark outputs to HDFS
* Collaborating through GitHub branches, commits, and pull requests
* Preparing technical evidence for report and presentation

---

## Acknowledgements

The team would like to thank:

* University of Economics Ho Chi Minh City
* School of Technology and Design
* Big Data Course
* Apache Hadoop
* Apache Spark
* Python and PySpark documentation
* GitHub

---

## License

This project was developed for educational and academic purposes.

---

<div align="center">

Made with care by Group 2 – Walmart Big Data Analytics

</div>