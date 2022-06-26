## TRABAJO FINAL DATA SCIENCE - UNIVERSITAT DE BARCELONA
##
## Integrantes: Joan Orellana Rios - Lucio Fabián Bolatti Fernández - Oriol Gilbert

En este GitHub Page describimos nuestro trabajo final de curso realizado. El mismo se base en el RecSys Challenge 2022 (http://recsyschallenge.com/2022), usando los mismos data sets y scorings del challenge.


### Primer Análisis de los Datos e Intento de Armado de Arbol de Categorías

Comenzamos realizando un primer análisis de los datos, a partir del cual decidimos intentar realizar un Arbol de Categorías.

Dado que el trabajo abordado se trata de sesiones de compras de diversos productos, y categorías de estos productos, pensamos en los arboles de categorías que se suelen encontrar en la mayoría de las páginas de compras. De esta manera, si lográramos recontruir dicho arbol de categorías a partir de los datos, se tendría una buena categorización de los items, la cual se podría utilizar en el entrenamiento de un modelo.

Dicho proceso y su correspondiente documentación se puede encontrar en el código "Intento de Arbol de Clasificación - Documentado" en el repositorio de este trabajo.

Lamentablemente no se otubieron buenos resultados con este proceso, por lo cual se decidió descartarlo y proceder a realizar un Clustering.


### Análisis de las Componentes Principales (PCA)

Previo a la realización del Clustering, se realiza un Análisis de las Componenetes Principales (PCA por sus siglas en inglés). Esto permite reducir la dimensionalidad de los datos, es decir sus características (features). De esta manera se logra reducir la complexidad del problema (para el algoritmo) y los tiempos de cómputo.

El resultado obtenido fue de 16 componentes las cuales consiguen mantener un porcentaje de varianza explicada acumulada del casi el 90%.

El proceso realizado en detalle y correspondiente documentación se puede en el código "PCA_Seleccion_Numero_Componentes" en el repositorio de este trabajo.



### Clustering

A partir del resultado del PCA (16 componenetes principales) de realiza un clustering de los datos con diversos métodos.

Luego de varios intentos y análisis, se obtuvo como mejor resultado un total de 4 clusters, que se utilizan para los posteriores pasos.

El detalle de esto y su correspondiente documentación se puede en el código "clustering_features" en el repositorio de este trabajo.


### Feature Engineering

Además de la información proporcionada por el Clustering, se realizó un Feature Engineering para obtener otra información relevante para el entrenamiento del modelo.

En primer lugar se tiene información determinada a criterio de los integrantes del grupo, y se trata de la siguente:

- Primer producto visto en la sesión
- Momento en el que el primer producto es visto
- Último producto visto en la sesión
- Momento en el que el último producto es visto
- Duración total de la sesión
- Tiempo promedio utilizado en cada producto
- Período del día en el cual la sesión comenzó (madrugada, día, tarde o noche)


#### Purchases

Con respecto a los datos de las compras, se extrajo el item comprado y la hora de la compra.

#### Features

Por otra parte, de cada sesión se determinó:

- Veces que se vió cada cluster
- Cluster más visto

#### Combinación de todos los datos

Por último, se combió todo lo anterior en un mismo dataset que, luego de algunos procesamientos más, es el utilizado como información de entrada para el entrenamiento del modelo.


### División en Train y Test

Previo a realizar los siguientes procesamiento de los datos, se realiza la división en train y test.


### Preparación de los Datos para los Algoritmos de Machine Learning

Para poder utilizar el dataset en el entramiento del modelo, es necesario realizar una serie de procesamientos con las variables categóricas, para lo cual se realizó un Ordinal Encoder y un Pipeline

COREGIRRRRRRRRRRRRRRRRR

### Entrenamiento y Selección del Modelo

Una vez listo el data set, se pasó al entramiento y selección del mejor modelo. Para ello se entrenaron divsersos modelos, los cuales son:

- SGD CLassifier
- Decision Tree
- K-nearest Neighbour
- SVC
- Random Forest Classifier

Una vez entrenados todos los modelos listados, se utiliza el método de la media recíproca para determinar cual es el mejor, ya que este es el método utilizado en el chalenge.

Como resultado, se determina que el mejor modelo a utilizar es el Random Forest Classifier, y con el cual se trabajó en los pasos posteriores.

### Mejores Parámetros

VER QUE PONER ACAAAAAA

### Feature Importance

VER QUE PONER ACAAAAA

### Generación del archivo Test Submission

Por último, utilizando el modelo descripto y todo lo anterior, se analiza el dataset de "test_final", y se genera el archivo de entrega "final_submission", el cual es el resultado final de este trabajo.


Todo esto en detalle y su correspondiente documentación se puede en el código "ALL_Joan" en el repositorio de este trabajo.




---------------------------------------------------------------------------------------------------------------



Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/LucioBolatti/RecSys22-UB/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
