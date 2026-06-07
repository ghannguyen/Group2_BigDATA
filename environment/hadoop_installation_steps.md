# Hadoop Installation Steps - Nhom_02

## 1. Environment

- OS: macOS Apple Silicon M1
- Shell: zsh
- Hadoop version: 3.4.3
- HDFS address: hdfs://localhost:9000
- Hadoop mode: Single Node / Pseudo Distributed Mode

## 2. Check Java

Command:

    java -version
    /usr/libexec/java_home -V

Result:

    JDK 17 is used for Hadoop and Spark.

## 3. Download Hadoop

Command:

    mkdir -p ~/hadoop
    cd ~/hadoop
    curl -O https://dlcdn.apache.org/hadoop/common/hadoop-3.4.3/hadoop-3.4.3.tar.gz
    tar -xzf hadoop-3.4.3.tar.gz

## 4. Configure environment variables in ~/.zshrc

Command:

    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export HADOOP_HOME=$HOME/hadoop/hadoop-3.4.3
    export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
    export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

Reload:

    source ~/.zshrc

## 5. Check Hadoop

Command:

    which hadoop
    hadoop version

## 6. Create local HDFS directories

Command:

    mkdir -p ~/hadoop_data/hdfs/namenode
    mkdir -p ~/hadoop_data/hdfs/datanode
    mkdir -p ~/hadoop_tmp

## 7. Configure Hadoop XML files

Configuration files:

    core-site.xml
    hdfs-site.xml
    mapred-site.xml
    yarn-site.xml

## 8. Configure SSH localhost

Command:

    ssh localhost

## 9. Format NameNode

Command:

    hdfs namenode -format

Expected result:

    Storage directory /Users/hannguyen/hadoop_data/hdfs/namenode has been successfully formatted.

## 10. Start Hadoop

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

## 11. Hadoop Web UI

URLs:

    NameNode UI: http://localhost:9870
    DataNode UI: http://localhost:9864
    YARN ResourceManager UI: http://localhost:8088
