# Clasificación y Generación Automática de Letras de Canciones

Este proyecto implementa un pipeline integral de Data Science aplicado al análisis y procesamiento de letras de canciones. Se abordan técnicas tradicionales de Machine Learning, Deep Learning y modelos de Transformer, cubriendo desde el análisis exploratorio y preprocesamiento del texto, hasta la clasificación, generación de respuestas, resumen y hasta una demostración básica de síntesis de voz y melodía.

## Índice

- [Introducción](#introducción)
- [Descripción del Dataset](#descripción-del-dataset)
- [Objetivos del Proyecto](#objetivos-del-proyecto)
- [Pipeline del Proyecto](#pipeline-del-proyecto)
  - [1. Análisis Exploratorio de Datos (EDA)](#1-análisis-exploratorio-de-datos-eda)
  - [2. Preprocesamiento del Texto](#2-preprocesamiento-del-texto)
  - [3. Feature Engineering y Modelado Clásico](#3-feature-engineering-y-modelado-clásico)
  - [4. Clasificación con Deep Learning](#4-clasificación-con-deep-learning)
  - [5. Modelos Avanzados con Transformers y Otras Tareas](#5-modelos-avanzados-con-transformers-y-otras-tareas)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Instrucciones de Ejecución](#instrucciones-de-ejecución)
- [Conclusiones y Extensiones Futuras](#conclusiones-y-extensiones-futuras)
- [Autores](#autores)
- [Referencias](#referencias)

---

## Introducción

El presente proyecto tiene como finalidad desarrollar un sistema capaz de clasificar y generar letras de canciones mediante diversas técnicas de procesamiento de lenguaje natural (NLP). Se utiliza un pipeline de Data Science que abarca desde el análisis exploratorio de datos (EDA) y preprocesamiento del texto, pasando por la ingeniería de características para el entrenamiento de modelos clásicos de Machine Learning, hasta la aplicación de Deep Learning y modelos basados en Transformers para tareas más avanzadas como la generación de texto, respuesta a preguntas (QA) y resumen.

---

## Descripción del Dataset

El dataset utilizado proviene de la plataforma **Genius** y contiene información relevante sobre las canciones, incluyendo:
- **`lyrics`**: el texto completo de la letra de la canción.
- **`tag`**: etiqueta o categoría asignada a cada canción (el problema se enfoca en la clasificación multiclase, con 3 clases).
- **`artist`**: el nombre del artista o grupo musical.

Cada clase cuenta con aproximadamente 120,000 registros. Para propósitos de procesamiento y experimentación, se ha realizado una reducción mediante muestreo.
Se trata de un dataset muy pesado, así que se ha adaptado el código para que, según las capacidades del entorno de ejecución, se pueda decidir que porcentaje del dataset se usa.
Recalcar que esta versión reducida del dataset estará balanceada por clases para que el rendimiento de los modelos no disminuya.

LINK KAGGLE: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information

Se adjunta una muestra del dataset en la carpeta data.

---

## Objetivos del Proyecto

- **Exploración y Comprensión del Dataset:**  
  Identificar la distribución de etiquetas, artistas y analizar características estadísticas (longitud de las letras).
  
- **Preprocesamiento Robusto:**  
  Limpiar y normalizar las letras eliminando ruido (URLs, etiquetas HTML, puntuación) y obtener una versión tokenizada y lematizada del texto.

- **Ingeniería de Características y Modelado Clásico:**  
  Transformar el texto en una representación numérica mediante TF-IDF y entrenar un modelo tradicional (Multinomial Naive Bayes) optimizado con GridSearchCV, evaluado a través de métricas y una learning curve.

- **Clasificación con Deep Learning:**  
  Convertir el texto preprocesado en secuencias numéricas y entrenar un modelo LSTM que clasifique las canciones en las tres categorías definidas.

- **Tareas Avanzadas:**  
  Explorar modelos basados en Transformers para clasificación, QA, resumen y generación automática de letras, así como una integración básica de síntesis de voz y generación de melodías.

- **Comparación y Evaluación:**  
  Comparar los resultados obtenidos por los enfoques tradicionales y de Deep Learning, analizando su capacidad de generalización y viabilidad para despliegue en producción.

---

## Pipeline del Proyecto

### 1. Análisis Exploratorio de Datos (EDA)

En esta fase se realiza:
- **Carga y verificación:**  
  Se carga el dataset, se visualizan las primeras filas y se examina la estructura del DataFrame, detectando valores nulos y analizando los tipos de datos.
- **Distribución de etiquetas y artistas:**  
  Se generan conteos y gráficos (countplot, gráfico de barras) que muestran el número de canciones por categoría y los artistas más representados.
- **Análisis de la longitud de las letras:**  
  Se calcula el número de palabras por letra y se visualiza mediante histogramas y boxplots para detectar outliers y entender la variabilidad del texto.
- **Visualización del vocabulario:**  
  Se crea una WordCloud y se muestra un análisis de la frecuencia de palabras, proporcionando una visión general del contenido textual.
  - **Análisis de sentimiento:**
   Se evalúa la positividad (valence) de cada canción y se comparan los valores promedio de sentimiento entre géneros y artistas.
 
  - **Modelado de temas:**
    Utilizando Latent Dirichlet Allocation (LDA), implementado con la biblioteca Gensim, se identifican los temas más relevantes en los géneros y artistas analizados.
 
  - **Embeddings:** se crea y entrena un modelo Word2Vec para convertir palabras en vectores. A partir de estos embeddings, se generan representaciones vectoriales para las canciones y los artistas, lo que permite identificar cuáles son más similares entre sí. Posteriormente, se reduce la dimensionalidad de estos vectores utilizando t-SNE, lo que permite visualizar las canciones y artistas en un espacio bidimensional."

### 2. Preprocesamiento del Texto

La fase de preprocesamiento se centra en:
- **Limpieza del texto original (`lyrics`):**  
  Se eliminan URLs, etiquetas HTML, contenido entre corchetes y caracteres de puntuación, y se convierte el texto a minúsculas.
- **Tokenización y lematización:**  
  Se implementan dos métodos: uno basado en NLTK y otro (opcional) usando spaCy. Se eliminan stopwords y tokens muy cortos para obtener una versión limpia almacenada en `clean_lyrics`.
- **Análisis posterior:**  
  Se calcula la longitud del texto y se analiza la frecuencia de palabras para asegurar que el preprocesamiento ha sido efectivo.

### 3. Feature Engineering y Modelado Clásico

Esta sección transforma el texto limpio en características numéricas y utiliza modelos tradicionales:
- **Vectorización:**  
  Se utiliza TF-IDF (con un rango de n-gramas) para convertir `clean_lyrics` en una matriz numérica.
- **División del dataset:**  
  Se realiza una división estratificada en conjuntos de entrenamiento y prueba para mantener la distribución de etiquetas.
- **Entrenamiento de un modelo clásico:**  
  Se entrena un clasificador Multinomial Naive Bayes y se evalúa su desempeño mediante classification report, matriz de confusión y learning curve.
- **Optimización del Pipeline:**  
  Se utiliza GridSearchCV para ajustar hiperparámetros y encontrar la mejor configuración en el pipeline combinado de TF-IDF y el clasificador.

### 4. Clasificación con Deep Learning

El enfoque deep learning incluye:
- **Tokenización y Padding:**  
  Se usa Keras Tokenizer para convertir el texto preprocesado en secuencias numéricas de longitud fija, aplicando padding para que todas las secuencias tengan el mismo tamaño.
- **Codificación de Etiquetas:**  
  Se convierten las etiquetas de texto en valores numéricos mediante LabelEncoder.
- **Definición del Modelo LSTM:**  
  Se construye un modelo LSTM simple (con Embedding, LSTM y una capa densa final con softmax) para clasificar las letras en tres clases.
- **Entrenamiento y Evaluación:**  
  Se entrena el modelo y se monitorean las curvas de loss y accuracy, además de evaluarlo con una matriz de confusión y reporte de clasificación.

### 5. Modelos Avanzados con Transformers y Otras Tareas

El proyecto también incluye:
- **Transfer Learning para Clasificación:**  
  Se utiliza un modelo preentrenado (DistilBERT) para clasificación, mostrando cómo los modelos de Transformer pueden mejorar el desempeño.
- **Question Answering (QA):**  
  Se entrena un modelo de QA para responder preguntas basadas en las letras, utilizando la librería Transformers.
- **Summarization:**  
  Se genera un resumen automático de las letras con un modelo preentrenado de summarization (DistilBART).
- **Generación de Texto y Audio:**  
  Se emplea GPT-2 para generar letras imitando un estilo específico y se demuestra una integración básica de síntesis de voz y generación de melodía.

---

## Requisitos e Instalación

### Requisitos

- **Python 3.x**
- **Bibliotecas:**  
  - **Datos y Visualización:** Pandas, NumPy, Matplotlib, Seaborn, WordCloud  
  - **Preprocesamiento:** NLTK (y opcionalmente spaCy con `en_core_web_sm`)  
  - **Modelado Tradicional:** Scikit-learn  
  - **Deep Learning:** TensorFlow (Keras)  
  - **Transformers y QA:** Hugging Face Transformers, Datasets, PyTorch  
  - **Audio (TTS y melodía):** pyttsx3, pydub

### Instalación

Para instalar las dependencias necesarias, crea un conda environment usando el fichero YAML de configuración:

```bash
conda env create -f env.yml
