DESCRIPCIÓN DEL CÓDIGO IMPLEMENTADO:

En el siguiente ejercicio se aplican algoritmos de clustering de la biblioteca de Spark ML sobre un conjunto de datos, y se analiza el rendimiento de los mismos y los resultados.
El conjunto de datos forma parte de Databricks Dataset 1, el repositorio de conjuntos de datos de ejemplo de Databricks, por lo que no será necesario cargarlo de manera externa al implementar dicho código en el Cloud de Databricks.

Descripción de los datos usados:
Se trata de un conjunto de datos de dígitos manuscritos.
• Se encuentra dividido en dos particiones: train (60.000 ejemplos) y test (10.000 ejemplos).
• Los atributos son todos y cada uno de los niveles de intensidad de una imagen de 28 x 28 pixels.
• Tiene 10 clases (0, 1, 2, ..., 9).

El código se va a dividir en las siguientes tareas:

a) Inicialmente nos quedamos con el subconjunto de instancias cuya clase es {0, 1}:
1. Cargar el conjunto de datos.
2. Explorar el DataFrame.
3. Obtener un DataFrame reducido. Se utiliza el método filter para quedarnos sólo con aquellos casos en los que label sea igual a 0 o label sea igual a 1.
4. Realizar Clustering usando la columna “features”, ignorando la columna label.
   • Prueba Kmeans con varios valores distintos para k (por ejemplo entre 2 y 10).
   • Encuentra cual es el mejor valor de k usando “Silhouette score”.
5. Interpretar los resultados.

b) Se repite lo anterior quedándonos con el subconjunto de instancias cuya clase es {0, 8, 7, 1}.

NOTA: El código está escrito en los ficheros "Héctor J Aparicio CLUSTERING.html" y "Héctor J Aparicio CLUSTERING.scala", siendo el de extensión .html el código escrito y ejecutado en Databricks y el de extensión .scala la exportación de dicho código a fichero de texto.

NOTA: Todo el código implementado está convenientemente comentado para su mejor comprensión.

