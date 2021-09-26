// Databricks notebook source
// PRÁCTICA 2: ANÁLISIS CLÚSTER
//                    Héctor Jesús Aparicio Muñoz
//                    Asignatura: APRENDIZAJE NO SUPERVISADO
//                    Máster Universitario en Inteligencia de Negocio y Big Data en Entornos Seguros

// Vamos a aplicar en esta práctica algoritmos de clustering sobre un conjunto de datos, para analizar su rendimiento y los resultados obtenidos.

// COMMAND ----------

// El conjunto de datos que utilizaremos forma parte de Databricks Dataset. Si listamos todos los subdirectorios donde
// se encuentran los Datasets obtenemos:
dbutils.fs.ls("/databricks-datasets").foreach(println)

// COMMAND ----------

// El conjunto de datos se encuentra en el subdirectorio dbfs:/databricks-datasets/mnist-digits/
// Dentro del subdirectorio mnist-digits tenemos:
dbutils.fs.ls("/databricks-datasets/mnist-digits").foreach(println)

// COMMAND ----------

// Veamos el contenido del archivo README.md
val readme = sc.textFile("/databricks-datasets/mnist-digits/README.md")
readme.collect.foreach(println)

// COMMAND ----------

// Dentro de mnist-digits/data-001 tenemos:
dbutils.fs.ls("/databricks-datasets/mnist-digits/data-001").foreach(println)

// COMMAND ----------

// TAREA 1
// Vemos que el conjunto de datos se encuentra dividido en dos particiones: train y test.
// Vamos a cargar cada una de las dos particiones del conjunto de datos:
val train = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
val test = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")

// COMMAND ----------

// TAREA 2
// Vamos a explorar los DataFrames de datos:
train.show(5)
train.printSchema()
println("Tamaño del conjunto de datos de entrenamiento: " + train.count() + " registros.")
test.show(5)
test.printSchema()
println("Tamaño del conjunto de datos de test: " + test.count() + " registros.")

// COMMAND ----------

// El conjunto de datos de entrenamiento tiene 60000 registros, y el de test 10000 registros como hemos podido comprobar.
// Además vemos que hay una discrepancia en el número de elementos de la columna "features", ya que en el conjunto de entrenamiento
// aparecen 780 por registro mientras que en el de test parecen 778. En realidad deberían ser 784 en los dos casos, ya que según
// el enunciado cada registro representa cada uno de los niveles de intensidad de una imagen de 28 x 28 = 784 píxeles.

// De todas formas nos piden utilizar sólo el conjunto de datos de entrenamiento para la realización de esta práctica, así que
// a partir de ahora sólo trabajaremos con el DataFrame de entrenamiento "train".

// COMMAND ----------

// Veamos ahora el número de clases diferentes que hay y cuántos registros pertenecen a cada clase:

import org.apache.spark.sql.functions._   // Para poder usar métodos dentro de agg

val trainAllClases = train.groupBy("label")      // Agrupamos por clase
                          .agg(count("label"))   // Contamos el número de registros por clase
                          .orderBy("label")      // Ordenamos por clase
trainAllClases.show()

// COMMAND ----------

// Vamos a visualizar uno de los registros del DataFrame, en este caso el primero de ellos:
train.take(1)

// COMMAND ----------

// Por lo tanto, como hemos comprobado el DataFrame con los datos tiene dos columnas, una llamada "label" que nos indica la clase del dígito
// que representa el registro, que será un número entre 0 y 9, y otra columna llamada "features" en la que aparecen las intensidades en
// cada uno de los píxeles en que está dividido el dígito manuscrito que representa cada registro (que en ese caso vemos que son 780 píxeles).
// A continuación sólo aparecen los píxeles en los que la intensidad es distinta de cero, mediante dos arrays de valores, uno representa dichos
// píxeles y el otro el valor de intensidad de cada píxel.
// Se corresponde por lo tanto la estructura de los elementos de la columna "features" con la estructura de vectores dispersos (Sparse Vectors).

// COMMAND ----------

// TAREA 3
// Vamos a obtener ahora un DataFrame reducido, quedándonos con los registros en los que "label" sea igual a 0 ó 1:

import spark.implicits._   // Para poder usar la notación-$

val trainReducido = train.filter($"label" === 0.0 or $"label" === 1.0)
trainReducido.show(6)

// Si volvemos a calcular para este DataFrame reducido el número de registros por clase tenemos:
val trainReducidoAllClases = trainReducido.groupBy("label")      // Agrupamos por clase
                                          .agg(count("label"))   // Contamos el número de registros por clase
                                          .orderBy("label")      // Ordenamos por clase
trainReducidoAllClases.show()

// COMMAND ----------

// TAREA 4
// Vamos a realizar el clustering de este DataFrame reducido con valores de clase 0 ó 1.
// Para ello utilizaremos solamente la columna "features".

// La columna "features" se corresponde con la columna que espera el algoritmo de aprendizaje (en este caso K-Means), porque ya contiene
// la lista de valores que se usarán para realizar el agrupamiento.

// Voy a comprobar que es así, utilizando el Transformer VectorAssembler para obtener una nueva columna llamada "atributos", que veremos
// que es idéntica a la columna "features"
import org.apache.spark.ml.feature.VectorAssembler
val columnas_kmeans = Array("features")
val assembler = new VectorAssembler().setInputCols(columnas_kmeans).setOutputCol("atributos")
val atributosDF = assembler.transform(trainReducido)
atributosDF.show(6)

println(atributosDF.select("features").take(1)(0))
println(atributosDF.select("atributos").take(1)(0))

// COMMAND ----------

// Por lo tanto para realizar el clustering de los datos nos quedamos con el DataFrame "trainReducido", no hace falta utilizar "atributosDF".

// Importamos KMeans
import org.apache.spark.ml.clustering.KMeans

// Creamos un objeto de tipo KMeans, y fijamos los parámetros del modelo para que busque 2 clústeres y que opere sobre la columna "features"
// y genere las predicciones en la columna "prediction"
val kmeansK2 = new KMeans().setK(2).setFeaturesCol("features").setPredictionCol("prediction")

// Entrenamos a continuación el modelo de K-Means. Normalmente tendríamos un conjunto de datos de entrenamiento y otro de datos de test,
// o dividiríamos el conjunto de datos de entrenamiento en dos particiones, una para entrenar el modelo y otra para test.
// En esta práctica no nos piden que lo dividamos, por lo tanto usamos el conjunto de datos de entrenamiento al completo para entrenar
// el modelo:
val modelKMeansK2 = kmeansK2.fit(trainReducido)

// Aplicamos el modelo entrenado sobre el propio conjunto de datos de entrenamiento para realizar el clustering de los datos:
val prediccionesK2 = modelKMeansK2.transform(trainReducido)
prediccionesK2.show()

// COMMAND ----------

// Vamos a probar a k-Means con varios valores distintos de K entre 2 y 10:

// Para K = 3
val kmeansK3 = new KMeans().setK(3).setFeaturesCol("features").setPredictionCol("prediction")
val modelKMeansK3 = kmeansK3.fit(trainReducido)
val prediccionesK3 = modelKMeansK3.transform(trainReducido)
prediccionesK3.show(6)

// Para K = 4
val kmeansK4 = new KMeans().setK(4).setFeaturesCol("features").setPredictionCol("prediction")
val modelKMeansK4 = kmeansK4.fit(trainReducido)
val prediccionesK4 = modelKMeansK4.transform(trainReducido)
prediccionesK4.show(6)

// Para K = 7
val kmeansK7 = new KMeans().setK(7).setFeaturesCol("features").setPredictionCol("prediction")
val modelKMeansK7 = kmeansK7.fit(trainReducido)
val prediccionesK7 = modelKMeansK7.transform(trainReducido)
prediccionesK7.show(6)

// Para K = 10
val kmeansK10 = new KMeans().setK(10).setFeaturesCol("features").setPredictionCol("prediction")
val modelKMeansK10 = kmeansK10.fit(trainReducido)
val prediccionesK10 = modelKMeansK10.transform(trainReducido)
prediccionesK10.show(6)

// COMMAND ----------

// Vamos a evaluar el clustering realizado mediante el cómputo del "Silhouette score"

import org.apache.spark.ml.evaluation.ClusteringEvaluator
val evaluator = new ClusteringEvaluator()

// Para K = 2
val silhouetteK2 = evaluator.evaluate(prediccionesK2)
println(s"Silhouette calculado con distancia euclídea para el caso K = 2 es $silhouetteK2")

// Para K = 3
val silhouetteK3 = evaluator.evaluate(prediccionesK3)
println(s"Silhouette calculado con distancia euclídea para el caso K = 3 es $silhouetteK3")

// Para K = 4
val silhouetteK4 = evaluator.evaluate(prediccionesK4)
println(s"Silhouette calculado con distancia euclídea para el caso K = 4 es $silhouetteK4")

// Para K = 7
val silhouetteK7 = evaluator.evaluate(prediccionesK7)
println(s"Silhouette calculado con distancia euclídea para el caso K = 7 es $silhouetteK7")

// Para K = 10
val silhouetteK10 = evaluator.evaluate(prediccionesK10)
println(s"Silhouette calculado con distancia euclídea para el caso K = 10 es $silhouetteK10")

// COMMAND ----------

// "Silhouette score" es una medida de la calidad del agrupamiento, que puede valer entre -1 y 1. Cuanto mayor sea su valor mejor es el
// agrupamiento, es decir, cuanto más cerca esté de 1 significa que los datos están agrupados adecuadamente. Si el valor fuera muy bajo
// significaría que se ha seleccionado un número no adecuado de clústeres, demasiados o demasiados pocos.

// Según eso, y observando los resultados obtenidos para diferentes valores de K, podemos decir que el valor óptimo de K en ese caso es K = 2,
// ya que es para el que obtenemos el mayor valor de "Silhouette score", el cual va disminuyendo al ir aumentando el número de clústeres.

// COMMAND ----------

// TAREA 5
// Interpretación de los resultados.

// Vamos a comparar los grupos que hace K-Means con las clases reales.
// Predicciones para los valores de la clase 0:
val gruposClase0 = prediccionesK2.filter($"label" === 0.0)
                                 .groupBy("prediction")
                                 .agg(count("prediction"))
println("Predicciones para los valores de la clase 0:")
gruposClase0.show()

// Predicciones para los valores de la clase 1:
val gruposClase1 = prediccionesK2.filter($"label" === 1.0)
                                 .groupBy("prediction")
                                 .agg(count("prediction"))
println("Predicciones para los valores de la clase 1:")
gruposClase1.show()

// COMMAND ----------

// Calculemos la tasa de acierto y de error globales que hemos obtenido.

// NOTA: Saldrá una tasa de acierto muy alta debido a que hemos usado el mismo conjunto de datos para entrenar el modelo y para evaluar
// la calidad de las predicciones. Para obtener tasas de acierto y error reales deberíamos haber usado dos conjuntos de datos diferentes,
// uno de entrenamiento y otro de test.

// Para calcular la tasa de acierto global dividiré el número de registros en los que la predicción es igual a la clase dada, por el número
// total de registros:
val aciertos = prediccionesK2.filter("label == prediction").count
val totales = prediccionesK2.count
val tasaAcierto = aciertos/totales.toDouble

// Para calcular la tasa de error global, no tengo más que restar a 1 la tasa de acierto global:
val tasaError = 1 - tasaAcierto

println("La tasa de acierto global al realizar el clustering con K = 2 es " + tasaAcierto)
println("La tasa de error global al realizar el clustering con K = 2 es " + tasaError)
println

// COMMAND ----------

// Si calculamos esa tasa de acierto y error al agrupar las instancias de cada una de las clases obtenemos lo siguiente:

// NOTA: Saldrá una tasa de acierto muy alta debido a que hemos usado el mismo conjunto de datos para entrenar el modelo y para evaluar
// la calidad de las predicciones. Para obtener tasas de acierto y error reales deberíamos haber usado dos conjuntos de datos diferentes,
// uno de entrenamiento y otro de test.

// Para la clase 0:
val aciertosClase0 = prediccionesK2.filter($"label" === 0.0 and $"prediction" === 0).count
val totalesClase0 = prediccionesK2.filter($"label" === 0.0).count
val tasaAciertoClase0 = aciertosClase0/totalesClase0.toDouble
val tasaErrorClase0 = 1 - tasaAciertoClase0

// Para la clase 1:
val aciertosClase1 = prediccionesK2.filter($"label" === 1.0 and $"prediction" === 1).count
val totalesClase1 = prediccionesK2.filter($"label" === 1.0).count
val tasaAciertoClase1 = aciertosClase1/totalesClase1.toDouble
val tasaErrorClase1 = 1 - tasaAciertoClase1

println("Agrupación de instancias de la clase 0:")
println("\tTasa Acierto = " + tasaAciertoClase0)
println("\tTasa Error = " + tasaErrorClase0)
println("Agrupación de instancias de la clase 1:")
println("\tTasa Acierto = " + tasaAciertoClase1)
println("\tTasa Error = " + tasaErrorClase1)
println

// COMMAND ----------

// NOTA: Como ya dije sale una tasa de acierto muy alta debido a que hemos usado el mismo conjunto de datos para entrenar el modelo y para
// evaluar la calidad de las predicciones. Para obtener tasas de acierto y error reales deberíamos haber usado dos conjuntos de datos
// diferentes, uno de entrenamiento y otro de test.

// Vemos por lo tanto que en estas condiciones el agrupamiento realizado por el algoritmo K-Means para las instancias que corresponden a 0's y
// 1's es muy bueno, con una tasa de acierto global de más del 99%.
// Si nos fijamos en cómo agrupa las instancias de la clase 0, vemos que en más del 98% de los casos acierta agrupándolas en el clúster 0,
// lo que corresponde a 5809 registros sobre un total de 5923 registros de clase 0.
// Para el caso de las instancias de clase 1, vemos que acierta en más de un 99,9% de los casos agrupándolas en el clúster 1, lo que
// corresponde a 6736 registros sobre un total de 6742 registros de clase 1, sólo ha fallado en 6 registros.

// COMMAND ----------

// Visualización de la agrupación en clústeres:

// Se registra el DataFrame como una tabla temporal
prediccionesK2.registerTempTable("resultados_table")
// Visualizamos la tabla en Databricks con el siguiente comando
display(sqlContext.sql("select * from resultados_table"))

// COMMAND ----------

// TAREA OPCIONAL
// En el caso de que nos quedáramos con las clases 0, 8, 7, 1
val trainReducidoNuevo = train.filter($"label" === 0.0 or $"label" === 1.0 or $"label" === 7.0 or $"label" === 8.0)
trainReducidoNuevo.show(10)

// Si calculamos para este nuevo DataFrame reducido el número de registros por clase tenemos:
val trainReducidoNuevoAllClases = trainReducidoNuevo.groupBy("label")      // Agrupamos por clase
                                                    .agg(count("label"))   // Contamos el número de registros por clase
                                                    .orderBy("label")      // Ordenamos por clase
trainReducidoNuevoAllClases.show()

// COMMAND ----------

// Almaceno las siguientes variables con el número total de registros de cada una de las clases, ya que las usaré más adelante.
val numTotalRegistrosClase0 = trainReducidoNuevo.filter($"label" === 0.0).count()
val numTotalRegistrosClase1 = trainReducidoNuevo.filter($"label" === 1.0).count()
val numTotalRegistrosClase7 = trainReducidoNuevo.filter($"label" === 7.0).count()
val numTotalRegistrosClase8 = trainReducidoNuevo.filter($"label" === 8.0).count()

// COMMAND ----------

// Realizamos a continuación el clustering de los datos.

// Ya hemos importado KMeans anteriormente
//import org.apache.spark.ml.clustering.KMeans

// Creamos un objeto de tipo KMeans, y fijamos los parámetros del modelo para que busque 4 clústeres y que opere sobre la columna "features"
// y genere las predicciones en la columna "prediction"
val kmeansNuevoK4 = new KMeans().setK(4).setFeaturesCol("features").setPredictionCol("prediction")

// Entrenamos a continuación el modelo de K-Means:
val modelKMeansNuevoK4 = kmeansNuevoK4.fit(trainReducidoNuevo)

// Aplicamos el modelo entrenado sobre el propio conjunto de datos de entrenamiento para realizar el clustering de los datos:
val prediccionesNuevoK4 = modelKMeansNuevoK4.transform(trainReducidoNuevo)
prediccionesNuevoK4.show()

// COMMAND ----------

// Vamos a probar a k-Means con varios valores distintos de K:

// Para K = 2
val kmeansNuevoK2 = new KMeans().setK(2).setFeaturesCol("features").setPredictionCol("prediction")
val modelKMeansNuevoK2 = kmeansNuevoK2.fit(trainReducidoNuevo)
val prediccionesNuevoK2 = modelKMeansNuevoK2.transform(trainReducidoNuevo)
prediccionesNuevoK2.show(10)

// Para K = 3
val kmeansNuevoK3 = new KMeans().setK(3).setFeaturesCol("features").setPredictionCol("prediction")
val modelKMeansNuevoK3 = kmeansNuevoK3.fit(trainReducidoNuevo)
val prediccionesNuevoK3 = modelKMeansNuevoK3.transform(trainReducidoNuevo)
prediccionesNuevoK3.show(10)

// Para K = 5
val kmeansNuevoK5 = new KMeans().setK(5).setFeaturesCol("features").setPredictionCol("prediction")
val modelKMeansNuevoK5 = kmeansNuevoK5.fit(trainReducidoNuevo)
val prediccionesNuevoK5 = modelKMeansNuevoK5.transform(trainReducidoNuevo)
prediccionesNuevoK5.show(10)

// COMMAND ----------

// Evaluamos el clustering realizado mediante el cómputo del "Silhouette score"

// Anteriormente ya hemos importado ClusteringEvaluator y hemos generado una instancia del mismo
//import org.apache.spark.ml.evaluation.ClusteringEvaluator
//val evaluator = new ClusteringEvaluator()

// Para K = 2
val silhouetteNuevoK2 = evaluator.evaluate(prediccionesNuevoK2)
println(s"Silhouette calculado con distancia euclídea para el caso K = 2 es $silhouetteNuevoK2")

// Para K = 3
val silhouetteNuevoK3 = evaluator.evaluate(prediccionesNuevoK3)
println(s"Silhouette calculado con distancia euclídea para el caso K = 3 es $silhouetteNuevoK3")

// Para K = 4
val silhouetteNuevoK4 = evaluator.evaluate(prediccionesNuevoK4)
println(s"Silhouette calculado con distancia euclídea para el caso K = 4 es $silhouetteNuevoK4")

// Para K = 5
val silhouetteNuevoK5 = evaluator.evaluate(prediccionesNuevoK5)
println(s"Silhouette calculado con distancia euclídea para el caso K = 5 es $silhouetteNuevoK5")

// COMMAND ----------

// En este caso y a la vista de los valores obtenidos de "Silhouette scores" el número de clústeres óptimo entre los que hemos
// probado no es 4, como sería lo normal, sino 2.
// Eso puede que se deba a que agrupa juntos varios dígitos que se representan de manera similar. Vamos a verlo de todas formas a continuación
// y haremos una interpretación de los resultados. Haremos esto para los casos de agrupamiento en 2 y en 4 clústeres.

// COMMAND ----------

// Caso K = 2

// Vamos a comparar los grupos que hace K-Means con las clases reales.

// NOTA: Los números totales de registros de cada clase los hemos calculado con anterioridad.

// Predicciones para los valores de la clase 0:
val gruposK2Clase0 = prediccionesNuevoK2.filter($"label" === 0.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase0.toDouble)
println("Predicciones para los valores de la clase 0 cuando agrupamos en 2 clústeres:")
gruposK2Clase0.show()

// Predicciones para los valores de la clase 1:
val gruposK2Clase1 = prediccionesNuevoK2.filter($"label" === 1.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase1.toDouble)
println("Predicciones para los valores de la clase 1 cuando agrupamos en 2 clústeres:")
gruposK2Clase1.show()

// Predicciones para los valores de la clase 7:
val gruposK2Clase7 = prediccionesNuevoK2.filter($"label" === 7.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase7.toDouble)
println("Predicciones para los valores de la clase 7 cuando agrupamos en 2 clústeres:")
gruposK2Clase7.show()

// Predicciones para los valores de la clase 8:
val gruposK2Clase8 = prediccionesNuevoK2.filter($"label" === 8.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase8.toDouble)
println("Predicciones para los valores de la clase 8 cuando agrupamos en 2 clústeres:")
gruposK2Clase8.show()

// COMMAND ----------

// Visualización de la agrupación en 2 clústeres:

// Se registra el DataFrame como una tabla temporal
prediccionesNuevoK2.registerTempTable("resultados_tableK2")
// Visualizamos la tabla en Databricks con el siguiente comando
display(sqlContext.sql("select * from resultados_tableK2"))

// COMMAND ----------

// Vemos que al hacer el agrupamiento en 2 clústeres hace el siguiente agrupamiento:
//       En el clúster 0:   6,47% de los registros de la clase 0
//                        100,00% de los registros de la clase 1
//                         97,14% de los registros de la clase 7
//                         92.53% de los registros de la clase 8
//       En el clúster 1: 93,53% de los registros de la clase 0
//                         2,86% de los registros de la clase 7
//                         7,47% de los registros de la clase 8
// Por lo tanto la mayoría de registros de las clases 1, 7 y 8 los agrupa en el mismo clúster, y la mayoría de los de la clase 0 en el
// otro clúster. Esto puede ser debido a que encuentre más similitud entre los dígitos manuscritos 1, 7 y 8, por ejemplo porque tengan una
// representación más estrecha que un 0 manuscrito, cuya representación es más ancha.
// En el caso del 1 y del 7, que se supone que tendrán una representación más similar, la coincidencia del clúster es casi total como vemos,
// con el 100% y el 97,14% de los registros respectivamente en el mismo clúster (el clúster 0).

// COMMAND ----------

// Caso K = 4

// Vamos a comparar los grupos que hace K-Means con las clases reales.

// NOTA: Los números totales de registros de cada clase los hemos calculado con anterioridad.

// Predicciones para los valores de la clase 0:
val gruposK4Clase0 = prediccionesNuevoK4.filter($"label" === 0.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase0.toDouble)
println("Predicciones para los valores de la clase 0 cuando agrupamos en 4 clústeres:")
gruposK4Clase0.show()

// Predicciones para los valores de la clase 1:
val gruposK4Clase1 = prediccionesNuevoK4.filter($"label" === 1.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase1.toDouble)
println("Predicciones para los valores de la clase 1 cuando agrupamos en 4 clústeres:")
gruposK4Clase1.show()

// Predicciones para los valores de la clase 7:
val gruposK4Clase7 = prediccionesNuevoK4.filter($"label" === 7.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase7.toDouble)
println("Predicciones para los valores de la clase 7 cuando agrupamos en 4 clústeres:")
gruposK4Clase7.show()

// Predicciones para los valores de la clase 8:
val gruposK4Clase8 = prediccionesNuevoK4.filter($"label" === 8.0)
                                        .groupBy("prediction")
                                        .agg(count("prediction"))
                                        .withColumn("Porcentaje_%", col("count(prediction)")*100/numTotalRegistrosClase8.toDouble)
println("Predicciones para los valores de la clase 8 cuando agrupamos en 4 clústeres:")
gruposK4Clase8.show()

// COMMAND ----------

// Visualización de la agrupación en 4 clústeres:

// Se registra el DataFrame como una tabla temporal
prediccionesNuevoK4.registerTempTable("resultados_tableK4")
// Visualizamos la tabla en Databricks con el siguiente comando
display(sqlContext.sql("select * from resultados_tableK4"))

// COMMAND ----------

// Vemos que al hacer el agrupamiento en 4 clústeres hace el siguiente agrupamiento:
//       En el clúster 0:  8,88% de los registros de la clase 0
//                         2,21% de los registros de la clase 1
//                         2,79% de los registros de la clase 7
//                        86.52% de los registros de la clase 8
//       En el clúster 1:  1,25% de los registros de la clase 0
//                         0,19% de los registros de la clase 1
//                        88,91% de los registros de la clase 7
//                         1,81% de los registros de la clase 8
//       En el clúster 2:  0,37% de los registros de la clase 0
//                        97,60% de los registros de la clase 1
//                         7,76% de los registros de la clase 7
//                        10.24% de los registros de la clase 8
//       En el clúster 3: 89,50% de los registros de la clase 0
//                         0,00% de los registros de la clase 1
//                         0,54% de los registros de la clase 7
//                         1,44% de los registros de la clase 8
// Por lo tanto en el clúster 0 se agrupan la mayoría de registros de la clase 8, en el clúster 1 la mayoría de los de la clase 7,
// en el clúster 2 la mayoría de los de la clase 1, y en el clúster 3 la mayoría de los de la clase 0.
// En este caso, los valores con los que menos error ha cometido son los de la clase 1, ya que el 97,60% de los mismos los ha agrupado
// en el mismo clúster.
// Podemos ver además que existe un porcentaje no despreciable de valores que ha agrupado de manera errónea, de la siguiente manera:
//   El 8,88% de los valores de la clase 0 los ha agrupado en el clúster que tiene mayoría de registros de la clase 8.
//   El 7,76% de los valores de la clase 7 los ha agrupado en el clúster que tiene mayoría de registros de la clase 1.
//   El 10,24% de los valores de la clase 8 los ha agrupado en el clúster que tiene mayoría de registros de la clase 1.
// Esto puede ser debido a que encuentre similitud entre algunos dígitos manuscritos de las clases 7 y 8 con los de la clase 1. Y por otro
// lado que también encuentre similitud entre ciertos dígitos manuscritos de la clase 0 con los de la clase 8.

// COMMAND ----------


