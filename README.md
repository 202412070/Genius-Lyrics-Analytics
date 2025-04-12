# Clasificación de Letras de Canciones: 

Este proyecto implementa un pipeline completo de Data Science aplicado al análisis y clasificación de letras de canciones. Se parte desde el análisis exploratorio (EDA) y el preprocesamiento del texto, pasando por la ingeniería de características y el entrenamiento tanto de modelos de Machine Learning clásicos como de Deep Learning, para finalmente comparar y evaluar el desempeño de ambos enfoques.

## Índice

- [Introducción](#introducción)
- [Descripción del Dataset](#descripción-del-dataset)
- [Pipeline del Proyecto](#pipeline-del-proyecto)
  - [1. Análisis Exploratorio de Datos (EDA)](#1-análisis-exploratorio-de-datos-eda)
  - [2. Preprocesamiento del Texto](#2-preprocesamiento-del-texto)
  - [3. Feature Engineering y Modelado Clásico](#3-feature-engineering-y-modelado-clásico)
  - [4. Modelado con Deep Learning](#4-modelado-con-deep-learning)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Cómo Ejecutar el Notebook](#cómo-ejecutar-el-notebook)
- [Conclusiones y Extensiones Futuras](#conclusiones-y-extensiones-futuras)
- [Referencias](#referencias)

---

## Introducción

El objetivo de este proyecto es desarrollar un sistema para la clasificación automática de letras de canciones. El problema se enmarca en la clasificación de texto, en el que se asigna una etiqueta (o *tag*) a cada canción en función de su contenido. Para ello, se sigue un típico pipeline de Data Science que incluye:

- **Análisis Exploratorio de Datos (EDA):** Entender la estructura, distribución y características del dataset.
- **Preprocesamiento del Texto:** Limpiar y normalizar el contenido textual para mejorar la calidad de las representaciones.
- **Feature Engineering:** Convertir el texto en una representación numérica (por ejemplo, mediante TF-IDF).
- **Modelado Clásico (ML):** Entrenar modelos tradicionales, como Multinomial Naive Bayes, optimizados con pipelines y técnicas de validación.
- **Deep Learning (DL):** Desarrollar un modelo LSTM simple para clasificar las letras a partir de secuencias numéricas.
- **Comparación y Evaluación:** Analizar métricas (accuracy, curva de learning, matriz de confusión) para evaluar el desempeño y la capacidad de generalización de cada enfoque.

---

## Descripción del Dataset

El dataset utilizado contiene letras de canciones extraídas de la plataforma Genius y cuenta con las siguientes columnas:

- **`lyrics`:** Texto completo de la letra de la canción.
- **`tag`:** Etiqueta o categoría a la que pertenece la canción. El problema en este caso es de clasificación multiclase (3 clases).
- **`artist`:** Nombre del artista de la canción.
- **`year`:** Año de la canción
- **`views`:** Número de visualizaciones de la cancíon
El dataset fue ampliado o reducido (por ejemplo, mediante muestreo) para facilitar el procesamiento y experimentación, garantizando un número balanceado de registros por clase.

---

## Pipeline del Proyecto

### 1. Análisis Exploratorio de Datos (EDA)

En esta fase se realizan múltiples análisis con el fin de entender a fondo el dataset antes de aplicar modelos predictivos:

- **Carga y Verificación del Dataset:**  
  Se utiliza Pandas para cargar el CSV, visualizar las primeras filas, examinar la estructura del DataFrame, detectar valores nulos y determinar los tipos de datos.

- **Distribución de Etiquetas y Artistas:**  
  Se generan conteos y visualizaciones (como countplots y gráficos de barras) para saber cuántas canciones hay por etiqueta y cuáles son los artistas más frecuentes.

- **Análisis de la Longitud de las Letras:**  
  Se calcula el número de palabras en cada letra y se visualiza la distribución mediante histogramas y boxplots. Este análisis ayuda a determinar la longitud máxima que se debe usar en el preprocesamiento (por ejemplo, para el padding de secuencias en Deep Learning).

- **Visualización del Contenido Textual:**  
  Se crean una WordCloud que muestra visualmente las palabras más comunes en el corpus, permitiendo identificar rápidamente términos recurrentes y posibles sesgos en el lenguaje. También se crean varias WordCloud, basadas en el conteo de palabras o en TF-IDF, que muestran las palabras más importantes para cada artista o género.

- **Análisis de sentimiento:**
  Se evalúa la positividad (valence) de cada canción y se comparan los valores promedio de sentimiento entre géneros y artistas.

 - **Modelado de temas:**
   Utilizando Latent Dirichlet Allocation (LDA), implementado con la biblioteca Gensim, se identifican los temas más relevantes en los géneros y artistas analizados.

 - **Embeddings:** se crea y entrena un modelo Word2Vec para convertir palabras en vectores. A partir de estos embeddings, se generan representaciones vectoriales para las canciones y los artistas, lo que permite identificar cuáles son más similares entre sí. Posteriormente, se reduce la dimensionalidad de estos vectores utilizando t-SNE, lo que permite visualizar las canciones y artistas en un espacio bidimensional."

### 2. Preprocesamiento del Texto

El preprocesamiento es crucial para mejorar la calidad de las características extraídas del texto. Se abordan las siguientes tareas:

- **Limpieza del Texto:**  
  Se eliminan elementos irrelevantes como URLs, etiquetas HTML y contenido entre corchetes. Además, se convierten todos los textos a minúsculas para normalizar.

- **Tokenización y Lematización:**  
  Se utilizan dos enfoques:
  - **NLTK:** Se emplea `nltk.word_tokenize` junto con lematización mediante `WordNetLemmatizer`, eliminando stopwords y tokens cortos.
  - **spaCy (opcional):** Se ilustra cómo se podría usar spaCy para obtener una tokenización y lematización más robusta, aunque se opta por NLTK por cuestiones de tiempo de procesamiento.

- **Generación de la Columna `clean_lyrics`:**  
  El resultado del preprocesamiento se almacena en una nueva columna que se utilizará en las siguientes etapas del pipeline.

### 3. Feature Engineering y Modelado Clásico

En esta etapa se transforma el texto limpio en características numéricas y se entrenan modelos clásicos de Machine Learning:

- **Vectorización con TF-IDF:**  
  Se utiliza el vectorizador TF-IDF para convertir cada letra en un vector numérico. Se configuran parámetros como `max_features` y `ngram_range` para capturar tanto unigrams como bigrams.

- **División del Dataset:**  
  Se separa el dataset en conjuntos de entrenamiento y prueba de forma estratificada para mantener la distribución de clases.

- **Modelado Clásico:**  
  Se entrena un modelo de Multinomial Naive Bayes (adecuado para datos de frecuencia de términos) y se evalúa mediante reportes de clasificación y la matriz de confusión.

- **Optimización con Pipeline y GridSearchCV:**  
  Se implementa un pipeline que combina el vectorizador TF-IDF y el clasificador, y se utiliza GridSearchCV para ajustar hiperparámetros y buscar la mejor configuración.

- **Learning Curve:**  
  Se generan curvas de aprendizaje para visualizar cómo varía la performance con el tamaño del conjunto de entrenamiento y para detectar posibles problemas de overfitting o underfitting.

### 4. Modelado con Deep Learning

Esta sección se centra en implementar y entrenar un modelo de Deep Learning para la clasificación de texto:

- **Tokenización y Padding:**  
  Se usa el `Tokenizer` de Keras para convertir el texto preprocesado en secuencias de enteros y se aplica padding para obtener secuencias de longitud fija. Esto es fundamental para alimentar el modelo LSTM.

- **Codificación de Etiquetas:**  
  Se emplea `LabelEncoder` para transformar las etiquetas en valores numéricos.

- **Definición del Modelo LSTM:**  
  El modelo consiste en:
  - Una capa `Embedding` para transformar las palabras en vectores densos.
  - Una capa LSTM (en el ejemplo, simple) que captura las dependencias secuenciales del texto.
  - Una capa densa final con activación `softmax` para predecir la probabilidad de cada clase.
  
  Se pueden incluir técnicas de regularización como dropout para evitar el sobreajuste.

- **Entrenamiento y Evaluación:**  
  Se entrena el modelo con EarlyStopping para detener el proceso cuando la pérdida de validación no mejora. Se evalúan las curvas de entrenamiento y validación (accuracy y loss) y se muestra la matriz de confusión para interpretar los errores.

### Comparación y Conclusiones

El notebook final permite comparar el desempeño de los enfoques clásicos y de Deep Learning. Se discuten aspectos tales como:

- **Precisión y capacidad de generalización:**  
  Se analiza si los modelos se ajustan demasiado a los datos de entrenamiento o si son capaces de generalizar bien a datos nuevos.
- **Curvas de aprendizaje:**  
  La visualización de las curvas ayuda a detectar si el modelo experimenta overfitting o underfitting.
- **Viabilidad de despliegue:**  
  Se plantea una discusión sobre cuál modelo sería más adecuado para un entorno real, considerando tanto el rendimiento como la complejidad y los tiempos de entrenamiento.

---

## Requisitos e Instalación

### Requisitos

- **Python 3.x**  
- **Bibliotecas:** Pandas, NumPy, Matplotlib, Seaborn, NLTK, spaCy, scikit-learn, TensorFlow (Keras) y wordcloud.

### Instalación

Para instalar las dependencias necesarias, utiliza:

```bash
pip install pandas numpy matplotlib seaborn nltk spacy scikit-learn tensorflow wordcloud
python -m spacy download en_core_web_sm
