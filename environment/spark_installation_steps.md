# Spark Installation Steps - Nhom_02

## 1. Environment

- OS: macOS Apple Silicon M1
- Shell: zsh
- Spark AppName: Nhom_02_Walmart_Big_Data
- HDFS address: hdfs://localhost:9000

## 2. Install Spark

Command:

    brew install apache-spark

## 3. Configure Spark environment in ~/.zshrc

Command:

    export SPARK_HOME=/opt/homebrew/opt/apache-spark/libexec
    export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

## 4. Check Spark

Command:

    which spark-submit
    spark-submit --version

## 5. Install PySpark

Command:

    python3 -m pip install pyspark
    python3 -c "import pyspark; print(pyspark.__version__)"

## 6. Run Spark job

Command:

    cd /Users/hannguyen/Documents/Group2_BigDATA
    python3 src/02_read_from_hdfs.py

## 7. Spark UI

URL:

    http://localhost:4050

## 8. Spark output path

Path:

    hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched
