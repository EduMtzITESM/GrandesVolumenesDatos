# Grandes Volúmenes de datos

+ Estudiante: Eduardo Selim Martínez Mayorga

# Actividad 3: Aprendizaje supervisado y no supervisado

# Parte 1: Introducción teórica

## Machine Learning

**Aprendizaje Supervisado**: Este tipo de aprendizaje utiliza conjuntos de datos etiquetados donde cada ejemplo de entrada tiene una salida conocida y correcta. El algoritmo aprende a mapear las entradas a las salidas correctas mediante la observación de estos pares entrada-salida durante el entrenamiento. Su objetivo principal es construir un modelo que pueda predecir con precisión las etiquetas o valores de nuevos datos no vistos anteriormente. Se divide en dos categorías: clasificación (predecir categorías discretas como spam/no spam) y regresión (predecir valores continuos como precios de casas).

**Aprendizaje No Supervisado**: En este enfoque, el algoritmo trabaja con datos que no tienen etiquetas o respuestas conocidas, debiendo descubrir patrones ocultos y estructuras en los datos por sí mismo. El objetivo es encontrar representaciones útiles de los datos, agrupar elementos similares, reducir dimensionalidad o detectar anomalías sin guidance externa. Las técnicas más comunes incluyen clustering (agrupamiento), reducción de dimensionalidad, y detección de patrones, siendo especialmente valioso para explorar datos desconocidos y encontrar insights que no son evidentes a simple vista.

**Aprendizaje Semi-supervisado**: Esta metodología combina elementos del aprendizaje supervisado y no supervisado, utilizando una pequeña cantidad de datos etiquetados junto con una gran cantidad de datos sin etiquetar. El objetivo es aprovechar la abundante información no etiquetada para mejorar el rendimiento del modelo más allá de lo que sería posible usando solo los limitados datos etiquetados. Es particularmente útil cuando obtener etiquetas es costoso o requiere mucho tiempo, como en reconocimiento de imágenes médicas o procesamiento de lenguaje natural, donde se puede usar texto abundante sin anotar para mejorar modelos entrenados con pocos ejemplos etiquetados.

**Aprendizaje por refuerzo**: En este paradigma, un agente aprende a tomar decisiones óptimas en un entorno mediante la interacción directa, recibiendo recompensas o castigos basados en sus acciones. El objetivo es que el agente desarrolle una política de comportamiento que maximice la recompensa acumulada a largo plazo, aprendiendo qué acciones tomar en diferentes situaciones a través de prueba y error. No requiere ejemplos etiquetados, sino que aprende de las consecuencias de sus propias acciones, siendo fundamental en aplicaciones como juegos, robótica, sistemas de recomendación y control autónomo, donde el agente debe aprender estrategias óptimas para alcanzar objetivos específicos.

En PySpark (la API de Python para Apache Spark), el módulo `pyspark.ml` proporciona varios **algoritmos de aprendizaje supervisado**, tanto para **clasificación** como para **regresión**. Aquí tienes un resumen de los principales:

### Clasificación (`pyspark.ml.classification`)

1. **Logistic Regression**
   `LogisticRegression()`
   Ideal para clasificación binaria y multiclase (mediante softmax).

2. **Decision Tree Classifier**
   `DecisionTreeClassifier()`
   Árbol de decisión para clasificación.

3. **Random Forest Classifier**
   `RandomForestClassifier()`
   Ensamble de árboles; mejora precisión y reduce overfitting.

4. **Gradient-Boosted Tree (GBT) Classifier**
   `GBTClassifier()`
   Variante de boosting sobre árboles; muy potente para clasificación binaria.

5. **Naive Bayes**
   `NaiveBayes()`
   Modelo probabilístico, útil con texto y datos categóricos.

6. **Multilayer Perceptron Classifier**
   `MultilayerPerceptronClassifier()`
   Red neuronal multicapa; útil para clasificación no lineal.

7. **Linear Support Vector Machine (SVM)**
   `LinearSVC()`
   SVM para clasificación binaria (con margen suave).

### Regresión (`pyspark.ml.regression`)

1. **Linear Regression**
   `LinearRegression()`
   Modelo lineal clásico.

2. **Generalized Linear Regression (GLR)**
   `GeneralizedLinearRegression()`
   Ampliación del modelo lineal para diferentes distribuciones (Poisson, Gamma, etc.).

3. **Decision Tree Regressor**
   `DecisionTreeRegressor()`
   Árbol de decisión para regresión.

4. **Random Forest Regressor**
   `RandomForestRegressor()`
   Ensamble de árboles para regresión.

5. **Gradient-Boosted Tree Regressor**
   `GBTRegressor()`
   Boosting sobre árboles para regresión.

6. **Isotonic Regression**
   `IsotonicRegression()`
   Regresión no paramétrica, útil cuando se asume una relación monótonamente creciente o decreciente.


### Otros Componentes Importantes

* **Pipelines y Transformers** para preprocesamiento: `VectorAssembler`, `StringIndexer`, `StandardScaler`, etc.
* **Evaluadores**: `BinaryClassificationEvaluator`, `MulticlassClassificationEvaluator`, `RegressionEvaluator`.

---

En **PySpark**, los algoritmos de **aprendizaje no supervisado** están disponibles principalmente en el módulo `pyspark.ml.clustering` (para *clustering*) y `pyspark.ml.fpm` (para *pattern mining*). A continuación te presento los principales algoritmos no supervisados que puedes usar:

1. **K-Means**
   `KMeans()`
   Clustering basado en centros de masa. Requiere especificar el número de clusters `k`.

2. **Bisecting K-Means**
   `BisectingKMeans()`
   Variante jerárquica de K-Means, útil para estructuras de cluster en árbol.

3. **Gaussian Mixture Model (GMM)**
   `GaussianMixture()`
   Modelo probabilístico que asume que los datos provienen de una mezcla de distribuciones normales. Devuelve probabilidades de pertenencia.

4. **Latent Dirichlet Allocation (LDA)**
   `LDA()`
   Modelo probabilístico para análisis de tópicos en texto. Agrupa documentos por temas latentes.

5. **Power Iteration Clustering (PIC)**
   `PowerIterationClustering()`
   Algoritmo de clustering espectral escalable, basado en el análisis del grafo de similitud entre elementos.

6. **Frequent Pattern Growth (FP-Growth)**
   `FPGrowth()`
   Algoritmo eficiente para encontrar conjuntos frecuentes y generar reglas de asociación. Útil en análisis de cestas de mercado (*market basket analysis*).


Aunque no están en un módulo específico de "unsupervised", PySpark ofrece soporte para algunos algoritmos de reducción de dimensiones, útiles para tareas no supervisadas:

7. **Principal Component Analysis (PCA)**
   `PCA()`
   Reducción de dimensionalidad lineal, mantiene la varianza máxima.

8. **Truncated Singular Value Decomposition (SVD)**
   `TruncatedSVD()`
   Reducción lineal de dimensiones (en Spark solo disponible para matrices dispersas como `RowMatrix` en `pyspark.mllib.linalg.distributed`).

# Partes: 2. Selección de los datos, 3. Preparación de los datos, 4. Preparación del conjunto de entrenamiento y prueba y 5. Construcción de modelos de aprendizaje supervisado y no supervisado

El desarrollo de estas secciones se encuenta en el el jupyter notebook que se encuentra en este repositorio en la liga:

[Jupyter Notebook](./ProyectoEntrega3ML_A01795167.ipynb)

## Referencias

+ https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html
