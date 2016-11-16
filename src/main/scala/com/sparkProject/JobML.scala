package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel, feature}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}


/**
  * Created by dorian on 27/10/16.
  */
object JobML {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_4_5")
      .getOrCreate()

    var path = ""
    var file_parquet = ""
    var file_output = ""

    if(args.length == 0) {
      path = "/mnt/3A5620DE56209C9F/Dorian/Formation/3. MS BGD Telecom ParisTech 2016-2017/Période 1/Introduction au framework hadoop/spark/tp2_3/"
      file_parquet = "cleanedDataFrame.parquet"
      file_output = "trained_model.model"
    }
    else {
      for( i <- args.indices ) {
        if(args(i).equals("-p") && i+1 < args.length) path = args(i+1)
        if(args(i).equals("-i") && i+1 < args.length) file_parquet = args(i+1)
        if(args(i).equals("-o") && i+1 < args.length) file_output = args(i+1)
      }
    }

    // Loading file
    val df = spark.read.parquet(path + file_parquet)
    println("Number of planets: " + df.count())
    println("Number of features: " + df.columns.length)
    df.printSchema()

    // Assembling features inside a column vector
    val vector_assembler = new feature.VectorAssembler()
      .setInputCols(df.drop("koi_disposition", "rowid").columns.array)
      .setOutputCol("features")

    // Convertir la colonne des labels en binaire
    val label_indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("koi_disposition_indexed")

    // Split data in training and test set
    val Array(training_set, test_set) = df.randomSplit(Array(0.9, 0.1), 1234)

    // Initialisation de l'estimateur
    val lr = new LogisticRegression()
      .setLabelCol("koi_disposition_indexed")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setElasticNetParam(1.0)
      .setStandardization(true)

    // Chain indexers and classifier in pipeline
    val pipeline = new Pipeline()
      .setStages(Array(label_indexer, vector_assembler, lr))

    // Create search grid
    val range_tol = (0.0 to 6.0 by 0.5).map{(f: Double) => math.pow(10, -f)}
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(50, 80, 100))
      .addGrid(lr.tol, range_tol)
      .addGrid(lr.regParam, Array(0.005, 0.1, 1))
      .addGrid(lr.fitIntercept)
      .build()

    // Compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("koi_disposition_indexed")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // Create cross validator
    val cv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(training_set)

    // Make predictions on test set
    val predictions = cvModel.transform(test_set)

    // Evaluate accuracy of predictions
    val accuracy = evaluator.evaluate(predictions)
    println("Précision de la prédiction = " + accuracy)

    // Select example rows to display
    predictions.select("prediction", "koi_disposition_indexed", "features").show(5)

    // Get the best model
    val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val bestModel = bestPipelineModel.stages(2).asInstanceOf[LogisticRegressionModel]

    // Best model hyper-parameters
    println("Best model parameters:")
    println("Max iteration: " + bestModel.getMaxIter)
    println("Tolerance: " + bestModel.getTol)
    println("Alpha (elasticnetparam): " + bestModel.getElasticNetParam)
    println("Regularization (lambda): " + bestModel.getRegParam)
    println("Intercept: " + bestModel.getFitIntercept)

    // Save the best model
    bestModel.save(path + file_output)
  }
}