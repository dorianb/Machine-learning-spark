# Machine-learning-spark

Exoplanets classification with spark ml


La ligne de commande suivante permet de sousmettre un job spark pour l'entraînement du modèle de classification d'exoplanètes:
```{r, engine='bash', count_lines}
./spark-submit --conf spark.eventLog.enabled=true \
 --conf spark.eventLog.dir="/tmp" --driver-memory 2G --executor-memory 6G \
 --class com.sparkProject.JobML --master spark://dorian-N56VB:7077 \
 "/path_to_project/target/scala-2.11/tp_spark-assembly-1.0.jar" \
 "-p" "/path_to_input_and_output_files/" \
 "-i" "input_file.parquet" \
 "-o" "output_file.model"
 ```
 
 Plusieurs paramètres sont à renseigner:
 * -p <le chemin vers le dossier contenant le fichier d'entrée et le fhcihier de sortie>
 * -i <nom du fichier d'entrée au format parquet, correspond aux données d'entrainement nétoyées>
 * -o <nom fichier de sortie au format model, correspond aux modèle entrâiné à persister>
