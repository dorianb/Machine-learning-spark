package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /********************************************************************************
      *
      *        TP 1
      *
      *        - Set environment, InteliJ, submit jobs to Spark
      *        - Load local unstructured data
      *        - Word count , Map Reduce
      ********************************************************************************/



    // ----------------- word count ------------------------

    val df_wordCount = sc.textFile("/opt/spark-2.0.0-bin-hadoop2.6/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()


    /********************************************************************************
      *
      *        TP 2 : début du projet
      *
      ********************************************************************************/

    val path = "/mnt/3A5620DE56209C9F/Dorian/Formation/3. MS BGD Telecom ParisTech 2016-2017/Période 1/Introduction au framework hadoop/spark/tp2_3/"
    val file_input = "cumulative.csv"
    val file_output = "planets.csv"
    val df_planet = spark.read.option("comment", "#").option("header","true").csv(path + file_input)

    println("Number of planets: " + df_planet.count())
    println("Number of features: " + df_planet.columns.length)

    df_planet.show()

    val cols = df_planet.columns.slice(10, 20)
    df_planet.select(cols.head, cols.tail:_*).show()

    df_planet.printSchema()

    df_planet.groupBy("koi_disposition").count().show()

    val df_planet_cln = df_planet.filter("koi_disposition IN ('CONFIRMED', 'FALSE POSITIVE')")
    println("Before cleaning: " + df_planet.count())
    println("After cleaning: " + df_planet_cln.count())

    df_planet_cln.agg(countDistinct('koi_eccen_err1)).show()
    val df_planet_cln2 = df_planet_cln.drop("koi_eccen_err1")
    val df_planet_cln3 = df_planet_cln2.drop("index", "kepid", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_sparprov",
      "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_dce_delivname", "koi_parm_prov",
      "koi_limbdark_mod", "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name", "kepler_name",
      "koi_vet_date", "koi_pdisposition")

    println("Before cleaning useless columns: " + df_planet.columns.length)
    println("After cleaning useless columns: " + df_planet_cln3.columns.length)

    val columns_to_drop = df_planet_cln3.columns.filter{ case (column: String) =>
      df_planet_cln3.agg(countDistinct(column)).first().getLong(0) <= 1}
    val df_planet_cln4 = df_planet_cln3.drop(columns_to_drop:_*)

    println("Number of columns to drop: " + columns_to_drop.length)
    println("After cleaning single value columns: " + df_planet_cln4.columns.length)

    df_planet_cln4.describe("koi_disposition", "koi_prad", "koi_rmag", "koi_zmag").show()
    df_planet_cln4.na.fill(0)

    val df_labels = df_planet_cln4.select("rowid", "koi_disposition")
    val df_features = df_planet_cln4.drop("koi_disposition")

    val df_joined = df_features.join(df_labels, usingColumn = "rowid")
    df_joined.printSchema()

    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)

    val df_newFeatures = df_planet_cln4
      .withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
      .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

    df_newFeatures.select("rowid", "koi_ror_min", "koi_ror_max").take(5).foreach(x => println(x))

    df_newFeatures
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(path + file_output)
  }
}
