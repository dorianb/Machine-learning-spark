# Machine-learning-spark
Exoplanets classification with spark ml

```{r, engine='bash', count_lines}
./spark-submit --conf spark.eventLog.enabled=true \
 --conf spark.eventLog.dir="/tmp" --driver-memory 2G --executor-memory 6G \
 --class com.sparkProject.JobML --master spark://dorian-N56VB:7077 \
 "/mnt/3A5620DE56209C9F/Dorian/Formation/3. MS BGD Telecom ParisTech 2016-2017/Période 1/Introduction au framework hadoop/spark/tp2_3/tp_spark/target/scala-2.11/tp_spark-assembly-1.0.jar" \
 "-p" "/mnt/3A5620DE56209C9F/Dorian/Formation/3. MS BGD Telecom ParisTech 2016-2017/Période 1/Introduction au framework hadoop/spark/tp2_3/" \
 "-i" "cleanedDataFrame.parquet" \
 "-o" "trained_model.model"
 ```
