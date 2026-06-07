#!/bin/bash

set -e

echo "============================================================"
echo "Nhom_02 - Upload Walmart Store Sales Dataset to HDFS"
echo "============================================================"

HDFS_RAW_DIR="/bigdata/walmart/raw"
HDFS_PROCESSED_DIR="/bigdata/walmart/processed"
LOCAL_RAW_DIR="data/raw"

echo ""
echo "1. Current directory:"
pwd

echo ""
echo "2. Check local CSV files:"
ls -lh ${LOCAL_RAW_DIR}

echo ""
echo "3. Check Hadoop version:"
hadoop version

echo ""
echo "4. Create HDFS directories:"
hdfs dfs -mkdir -p ${HDFS_RAW_DIR}
hdfs dfs -mkdir -p ${HDFS_PROCESSED_DIR}

echo ""
echo "5. Upload raw CSV files to HDFS:"
hdfs dfs -put -f ${LOCAL_RAW_DIR}/train.csv ${HDFS_RAW_DIR}/
hdfs dfs -put -f ${LOCAL_RAW_DIR}/features.csv ${HDFS_RAW_DIR}/
hdfs dfs -put -f ${LOCAL_RAW_DIR}/stores.csv ${HDFS_RAW_DIR}/
hdfs dfs -put -f ${LOCAL_RAW_DIR}/test.csv ${HDFS_RAW_DIR}/

echo ""
echo "6. Check uploaded files:"
hdfs dfs -ls -h ${HDFS_RAW_DIR}

echo ""
echo "7. Check HDFS file sizes:"
hdfs dfs -du -h ${HDFS_RAW_DIR}

echo ""
echo "8. Preview train.csv from HDFS:"
hdfs dfs -cat ${HDFS_RAW_DIR}/train.csv | head || true

echo ""
echo "============================================================"
echo "UPLOAD COMPLETED SUCCESSFULLY"
echo "============================================================"
