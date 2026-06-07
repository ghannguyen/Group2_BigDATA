# HDFS Commands - Nhom_02

## 1. Start Hadoop services

Command:

    start-dfs.sh
    start-yarn.sh
    jps

Expected processes:

    NameNode
    DataNode
    SecondaryNameNode
    ResourceManager
    NodeManager
    Jps

## 2. Create HDFS directories

Command:

    hdfs dfs -mkdir -p /bigdata/walmart/raw
    hdfs dfs -mkdir -p /bigdata/walmart/processed

## 3. Upload raw CSV files to HDFS

Command:

    hdfs dfs -put -f data/raw/train.csv /bigdata/walmart/raw/
    hdfs dfs -put -f data/raw/features.csv /bigdata/walmart/raw/
    hdfs dfs -put -f data/raw/stores.csv /bigdata/walmart/raw/
    hdfs dfs -put -f data/raw/test.csv /bigdata/walmart/raw/

## 4. Check uploaded files

Command:

    hdfs dfs -ls -h /bigdata/walmart/raw
    hdfs dfs -du -h /bigdata/walmart/raw
    hdfs dfs -cat /bigdata/walmart/raw/train.csv | head

Expected result:

    Found 4 items
    features.csv
    stores.csv
    test.csv
    train.csv

## 5. Check processed output

Command:

    hdfs dfs -ls -h /bigdata/walmart/processed
    hdfs dfs -ls -h /bigdata/walmart/processed/walmart_sales_enriched

Expected result:

    _SUCCESS
    part-xxxxx.snappy.parquet

## 6. Stop Hadoop services

Command:

    stop-dfs.sh
    stop-yarn.sh
    jps
