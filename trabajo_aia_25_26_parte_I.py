#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2025-26
# PARTE I del trabajo práctico: Implementación de árboles de decisión 
#                               y random forests
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: MORÓN GARCÍA
# NOMBRE: PABLO 
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: VARGAS BELLIDO
# NOMBRE: JUAN MANUEL
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La
# discusión y el intercambio de información de carácter general con
# los compañeros se permite, pero NO AL NIVEL DE CÓDIGO. Igualmente el
# remitir código de terceros, OBTENIDO A TRAVÉS DE LA RED, o de
# cualquier otro medio, se considerará plagio.

# El objetivo principal del trabajo es reforzar de manera práctica
# los conceptos aprendidos en clase, para alcanzar una mayor
# comprensión de los mismos a través de la implementación que se
# pide. Se permite, si así se desea, el uso de herramientas de
# inteligencia artificial generativa que asistan en el desarrollo
# código, pero esta herramienta ha de usarse sólo como un asistente
# que facilite el trabajo, y en ningún caso se debe entregar un código
# que no se conozca en profundidad y con detalle. 

# Para asegurar que la evaluación del mismo está alineada con el
# objetivo descrito en el párrafo anterior, el trabajo ha de ser
# presentado ante el profesor, explicando con detalle y a nivel de
# código la implementación entregada, y será necesario demostrar total
# comprensión del código entregado. Si el trabajo se hace en grupo,
# ambos miembros del grupo deben poder explicar con detalle de código
# cualquier parte del trabajo.

# Cualquier plagio o entrega de código cuyo funcionamiento no se sea
# capaz de explicar con detalle, significará automáticamente la
# calificación de CERO EN LA ASIGNATURA para TODOS los estudiantes
# involucrados. Independientemente de OTRAS ACCIONES DE CARÁCTER
# DISCIPLINARIO que se pudieran tomar.
# *****************************************************************************************


# MUY IMPORTANTE: 
# ===============    
    
# * NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
#   Y ATRIBUTOS QUE SE PIDEN. ADEMÁS: NO HACERLO EN UN NOTEBOOK.

# * En este trabajo NO SE PERMITE USAR Scikit Learn, salvo donde se dice expresamente.
#   En particular, si se pide implementar algo, se refiere a implementar en python,
#   sin usar Scikit Learn.  
  
# * Se recomienda (y se valora especialmente) el uso eficiente de numpy. Todos 
#   los datasets se suponen dados como arrays de numpy. 

# * Hay que entregar este archivo con las implementaciones realizadas, junto con una versión
#   de este mismo archivo, en la que no aparezcan comentarios al código (con el nombre 
#   trabajo_aia_25_26_parte_I_sc.py) 

# * AL FINAL DEL ARCHIVO hay una serie de ejemplos a ejecutar que están comentados, y que
#   será lo que se ejecute durante la presentación del trabajo al profesor.
#   En la versión final (tanto la original, como la versión sin comentarios), 
#   descomentar esos ejemplos del final y no dejar ninguna otra ejecución de ejemplos. 



import math
import random
import numpy as np
from sklearn.model_selection import train_test_split



# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar (casi) todos los conjuntos de datos,
# basta con tener descomprimido el archivo datos-trabajo-aia.zip (en el mismo sitio
# que este archivo) Y CARGARLOS CON LA SIGUIENTE ORDEN:
    
from carga_datos import *    

# Como consecuencia de la línea anterior, se habrán cargado los siguientes 
# conjuntos de datos, que pasamos a describir, junto con los nombres de las 
# variables donde se cargan. Todos son arrays de numpy: 


# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre pasajeros del Titanic y si sobrevivieron o no. Es una versión 
#   restringida de este conocido dataset, con solo tres caracteristicas:
#   Pclass, IsFemale y Age. Se carga en las variables X_train_titanic, 
#   y_train_titanic, X_test_titanic e y_test_titanic.

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (0:republicano o 1:demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos (ver 
#   descripción en votos.py)


# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer. 
#   Ver descripcición en sikit learn.

  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos del dataset original. 
#   Los textos se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    


#  Además, en la carpeta datos/ se tienen los siguientes datasets, que
#  habrán de ser procesado y cargado (es decir, no se caragan directamente con
#  carga_datos.py).   
    
# * Un archivo credito.csv con datos sobre concesión de prestamos en una entidad 
#   bancaria, en función de: tipo de empleo, si ya tiene productos finacieros 
#   contratados, número de propiedades, número de hijos, estado civil y nivel de 
#   ingresos (cargarlo usando pd.read_csv en arrays de numpy X_credito e y_credito,
#   donde X_credito son las seis primeras columnas e y_credito la última).


# * Un archivo adultDataset.csv, con datos de personas para poder predecir si
#   alguien gana más o menos de 50000 dólares anuales, en función de una serie 
#   de características (para más detalles, ver https://archive.ics.uci.edu/dataset/2/adult)  
#   Más adelante se explica cómo cargar y procesar este conjunto de datos. 

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En la carpeta digitdata están todos los datos en archivos de texto. 
#   Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 




# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

#  >>>Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#   (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
#  (array([0, 1]), array([136, 229]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([34, 57]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------

def particion_entr_prueba(X, y, test=0.20):
    """
    Divide X e y en entrenamiento y prueba de forma aleatoria y estratificada.

    Devuelve:
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba
    """

    if test <= 0 or test >= 1:
        raise ValueError("El valor de test debe estar entre 0 y 1")

    indices_train = []
    indices_test = []

    clases = np.unique(y)

    for clase in clases:
        indices_clase = np.where(y == clase)[0]

        indices_clase = indices_clase.copy()
        np.random.shuffle(indices_clase)

        n_test = int(round(len(indices_clase) * test))

        
        indices_test.extend(indices_clase[:n_test])
        indices_train.extend(indices_clase[n_test:])

    indices_train = np.array(sorted(indices_train))
    indices_test = np.array(sorted(indices_test))

    return X[indices_train], X[indices_test], y[indices_train], y[indices_test]


#Test para probar la funcion de particion_entr_prueba
iris=load_iris()
X_iris=iris.data
y_iris=iris.target

X_train_iris,X_test_iris,y_train_iris,y_test_iris=particion_entr_prueba(X_iris,y_iris,test=0.2)

print("Proporción de clases en el conjunto original:")
print(np.unique(y_iris,return_counts=True))
print("\nProporción de clases en el conjunto de entrenamiento:")
print(np.unique(y_train_iris,return_counts=True))
print("\nProporción de clases en el conjunto de prueba:")
print(np.unique(y_test_iris,return_counts=True))

#muestra de 5 primeros ejemplos del conjunto de entrenamiento
print("\nPrimeros 5 ejemplos del conjunto de entrenamiento:", X_train_iris[:5])







# ===============================================
# EJERCICIO 2: IMPLEMENTACIÓN ÁRBOLES DE DECISIÓN
# ===============================================


# En este ejercicio pedimos implementar en python un algoritmo de aprendizaje para árboles 
# de decisión. Los árboles de decisión que trataremos serán árboles binarios, en los que
# en cada nodo interior se pregunta por el valor de un atributo o característica dada, 
# y si ese valor es mayor o menor que un valor umbral dado. Este es el mismo tipo de árbol 
# de decisión que se  manejan en Scikit Learn. 

# Se puede obtener información de este tipo de árboles en la entrada "Decision Trees"
# del manual de Scikit Learn. También en la práctica del Titanic hecha en clase.

# Se propone la implementación de un clasificador basado en árboles de
# de decisión, entrenado usando el algoritmo CART, similar al que implementa 
# la clase DecisonTree de Scikit Learn, pero con ALGUNAS VARIANTES, que indicaremos más
# adelante.

# Los árboles de decisión están formados por nodos. Usar la siguiente clase para la
# implementación de los nodos:
    
class Nodo:
    def __init__(self, atributo=None, umbral=None, izq=None, der=None,distr=None,*,clase=None):
        self.atributo = atributo
        self.umbral = umbral
        self.izq = izq
        self.der = der
        self.distr= distr
        self.clase = clase
        
    def es_hoja(self):
        return self.clase is not None

# Pasamos a describir los distintos atributos de esta clase:

# - atributo: el atributo por el que se pregunta en el nodo. Referenciaremos a cada
#   atributo POR EL ÍNDICE DE SU POSICIÓN (el número de columna).
# - umbral: es el valor umbral por el que se pregunta en el nodo. Si la instancia tiene un
#   valor de atributo menor o igual que el umbral, se sigue por el subárbol izquierdo. En
#   caso contrario, por el subárbol derecho.
# - izq: es el nodo raiz del subárbol izquierdo.
# - der: el nodo raiz del subárbol derecho.
# - distr: es un diccionario cuyas claves son las posibles clases, y cuyos valores son
#   cuántos ejemplos del conjunto de entrenamiento correspondientes al nodo hay de cada
#   clase. Cuando decimos "ejemplos correspondientes al nodo" queremos decir aquellos que
#   cumplen todas las condiciones (desde la raiz) que llevan a ese nodo.
# - clase: Si el nodo es una hoja, es la clase que predice. Si no es una hoja, este valor es None.



# Lo que sigue es una descripción del algoritmo que se pide implementar para la
# construcción de un árbol de decisión. En principio describiremos la versión básica y más
# conocida, y posteriormente indicaremos las peculiaridades y variantes que pedimos
# introducir a esta versión básica.

# Supondremos que recibimos un conjunto de entrenamiento X,y y además dos valores max_prof
# y min_ejemplos_nodo_interior, que nos van a servir como condiciones adicionales para
# dejar de expandir un nodo. El algoritmo se define recursivamente y tiene además un
# argumento adicional prof (inicialmente 0), con la profundidad del nodo actual.  

# CONSTRUYE_ARBOL(X,y,min_ejemplos_nodo_interior,max_prof,prof=0):

# 1. SI prof es mayor o igual que max_prof, 
#       o el número de ejemplos de X es menor que min_ejemplos_nodo_interior,
#       o en X todos los ejemplos son de la misma clase:
#       ENTONCES:
#          Devolver un nodo hoja con la distribución de clases en X,
#                    y con la clase mayoritaria en X
# 2. EN OTRO CASO:
#        encontrar el MEJOR atributo A y el mejor umbral u para ese atributo
#        y particionar en dos tanto X como y:
#            * X_izq, y_izq los ejemplos cuyo valor de A es menor o igual que u
#            * X_der, y_der los ejemplos cuyo valor de A es mayor que u
#        Llamadas recursivas:
#            A_izq=CONSTRUYE_ARBOL(X_izq,y_izq,min_ejemplos_nodo_interior,max_prof,prof+1)
#            A_der=CONSTRUYE_ARBOL(X_der,y_der,min_ejemplos_nodo_interior,max_prof,prof+1)
#        Devolver un nodo interior con el atributo y umbral seleccionado,
#                 con la distribución de clases de X, y con A_izq y A_der
#                 como hijos izquierdo y derecho respectivamente.


# Lo anterior es la descripción básica. A continuación indicamos una serie de variantes y
# cuestiones adicionales que se le piden a esta implementación concreta:

# - Consideraremos la posibilidad de restringir los atributos a usar en el árbol a un
#   número de atributos dado n_atrs. Ese subconjunto de atributos se seleccionará
#   aleatoriamente al principio de la construcción del aŕbol y será el mismo para todos
#   los nodos.
#   Por ejemplo, si el dataset tiene 15 atributos y le damos n_atrs=9, al comienzo de la
#   construcción del árbol seleccionamos aleatoriamente 9 atributos, y ya en los nodos del
#   árbol solo podrán aparecer alguno de esos 9 atributos. Nótese que si n_atrs es igual
#   al total de atributos, tendríamos la versión estándar del algoritmo.
#   NOTA: téngase en cuenta que a diferencia de lo que ocurre en la versión clásica de
#   Random Forests, no sorteamos los atributos en cada nodo, sino que hay un único sorteo
#   inicial para todo el árbol.

# - A la hora de elegir el mejor atributo y umbral para la partición de los nodos
#   interiores, usar el criterio de mejor GANANCIA DE INFORMACIÓN (en particular, NO USAR GINI).

# - La principal carga computacional de este algoritmo se debe a la cantidad de candidatos a
#   mejor atributo y mejor umbral que hay que evaluar en cada nodo, para decidir cuál es
#   la mejor partición. El hecho de limitar el número de atributos candidatos (como se ha
#   descrito más arriba), va en esa dirección. 
#   Otra manera es limitar también los posibles valores umbrales a considerar
#   para cada atributo. Para ello, en la implementación que se pide actuaremos en dos
#   sentidos:
#      (a) Considerar solo como candidatos a umbral los puntos medios entre cada par de 
#         valores consecutivos del atributo en los que hay cambio de clase, para los
#         ejemplos correspondientes a ese nodo.
#         Por ejemplo, si ordenados los valores del atributo A en orden creciente, hay un
#         ejemplo con valor v1 de A y clase C1 y a continuación otro ejemplo con valor v2
#         en A y clase C2 distinta de C1, entonces (v1+v2)/2 es un posible valor umbral
#         candidato. El resto de valores NO se considera candidato.

#      (b) En cada nodo, para elegir los umbrales candidatos correspondientes a un atibuto,
#         no considerar todos los ejemplos que corresponden a ese nodo, sino 
#         sólo  una proporción de los mismos, seleccionada aleatoriamente. La proporción a
#         considerar se da en un parámetro prop_umbral.
#         Por ejemplo, si prop_umbral es 0.7 y el conjunto de ejemplos correspondientes al
#         nodo es de 200 ejemplos, entonces aplicaremos el proceso de selección de
#         umbrales candidatos descrito en (a) considerando sólo un suconjunto de 140
#         ejemplos seleccionado aleatoriamente de entre esos 200.  


def CONSTRUYE_ARBOL(X,y,atributos_candidatos,min_ejemplos_nodo_interior,max_prof,prop_umbral,prof=0):
    distr={c: np.sum(y==c) for c in np.unique(y)}
    clase_mayorit= np.bincount(y).argmax()

    if(prof>=max_prof or len(X)<min_ejemplos_nodo_interior or len(np.unique(y))==1):
        return Nodo(clase=clase_mayorit, distr=distr)
    
    #en otro caso, hay que encontrar el mejor atributo y umbral para la partición del nodo
    mejor_atributo, mejor_umbral = mejor_particion(X, y, atributos_candidatos, prop_umbral)

    #particionar en dos tanto X como y:
    mascara_izqd=X[:,mejor_atributo]<=mejor_umbral
    
    X_izq=X[mascara_izqd]
    y_izq=y[mascara_izqd]
    X_der=X[~mascara_izqd]
    y_der=y[~mascara_izqd]

    #Llamadas recursivas:
    nodo_izq=CONSTRUYE_ARBOL(X_izq,y_izq,atributos_candidatos,min_ejemplos_nodo_interior,max_prof,prop_umbral,prof+1)
    nodo_der=CONSTRUYE_ARBOL(X_der,y_der,atributos_candidatos,min_ejemplos_nodo_interior,max_prof,prop_umbral,prof+1)

    return Nodo(mejor_atributo,mejor_umbral,izq=nodo_izq,der=nodo_der,distr=distr)

#prop_umbral es la proporción de ejemplos a considerar para elegir los umbrales 
# candidatos. Por ejemplo, si prop_umbral es 0.7 y el conjunto de ejemplos 
# correspondientes al nodo es de 200 ejemplos, entonces aplicaremos el proceso
#  de selección de umbrales candidatos considerando sólo un suconjunto de 140
# ejemplos seleccionado aleatoriamente de entre esos 200.

#Definimos la funcion mejor_particion que encuentra el mejor atributo y umbral
# para la partición de los nodos interiores, usando el criterio de mejor GANANCIA 
# DE INFORMACIÓN, y considerando sólo como candidatos a umbral los puntos medios
def mejor_particion(X,y,atributos_candidatos,prop_umbral):
    mejor_ganancia=-1
    mejor_atributo=None
    mejor_umbral=None

    #Recorremos los atributos candidatos
    for atributo in atributos_candidatos:
        #Para cada atributo, obtenemos los umbrales candidatos usando la función
        # umbrales, que implementa las dos restricciones que se han descrito más 
        # arriba para limitar el número de umbrales candidatos.
        for umbral in umbrales(X[:, atributo], y,prop_umbral):
            mascara=X[:,atributo]<=umbral
            y_izq,y_der=y[mascara],y[~mascara]
            
            #Comparamos con 0 la ganancia de información de esta partición, para asegurarnos de que
            #esta partición mejora la clasificación respecto a no particionar, y además comprobamos que
            #hay ejemplos a ambos lados de la partición, para evitar particiones que no dividen el nodo.
            if(len(y_izq)==0 or len(y_der)==0):
                continue
            
            gan=ganancia(y,y_izq,y_der)
            if gan>mejor_ganancia:
                mejor_ganancia=gan
                mejor_atributo=atributo
                mejor_umbral=umbral
    return mejor_atributo, mejor_umbral

#definimos la función ganancia que calcula la ganancia de información de una
#partición dada por un atributo y un umbral, a partir de los conjuntos de ejemplos
def ganancia(y,y_izq,y_der):
    n=len(y)
    #la ganancia se calcula como la entropía de y menos la media ponderada de 
    # las entropías de y_izq e y_der, ponderada por el número de ejemplos que hay
    #  en cada uno de esos conjuntos
    return entropia(y) - (len(y_izq)/n)*entropia(y_izq) - (len(y_der)/n)*entropia(y_der)

def umbrales(X,y,prop_umbral):
    n=len(y)
    #Seleccionamos aleatoriamente una proporción de los ejemplos correspondientes al nodo
    n_muestras=max(1,int(round(n*prop_umbral)))
    indices_muestra=np.random.choice(n,size=n_muestras,replace=False)  
    X_muestra=X[indices_muestra]
    y_muestra=y[indices_muestra]
    
    #Ordenamos porque sólo consideramos como candidatos a umbral los puntos
    #  medios entre cada par de valores consecutivos del atributo en los que
    #  hay cambio de clase, para los ejemplos correspondientes a ese nodo.
    orden=np.argsort(X_muestra)
    X_ordenado=X_muestra[orden]
    y_ordenado=y_muestra[orden]

    candidatos_umbrales=[]
    for i in range(len(y_ordenado)-1):
        if y_ordenado[i] != y_ordenado[i+1]: #hay cambio de clase
            umbral=(X_ordenado[i]+X_ordenado[i+1])/2
            candidatos_umbrales.append(umbral)
    return candidatos_umbrales



def entropia(y):
    #Ponemos _ para indicar que no nos interesa el valor de las clases,
    # sino sólo sus frecuencias
    _, counts = np.unique(y, return_counts=True)
    # Calculamos la entropía
    probs = counts / len(y)
    #Ponemos log para que el resultado sea 0 cuando la probabilidad es 0, aunque en principio no debería haber probabilidades 0, porque sólo se consideran umbrales entre valores de clases distintas
    return -np.sum(probs * np.log2(probs))

#Prueba de construye arbol
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, stratify=y_iris, random_state=42
)

#Sacamos el número de atributos del dataset
atributos=list(range(X_train.shape[1]))

raiz=CONSTRUYE_ARBOL(X_train,y_train,atributos,min_ejemplos_nodo_interior=5,max_prof=3,prop_umbral=0.7)

print("¿Es hoja la raíz?", raiz.es_hoja())          # debe ser False
print("Atributo raíz:", raiz.atributo)               # algún número entre 0 y 3
print("Umbral raíz:", raiz.umbral)                   # algún float
print("Distribución raíz:", raiz.distr)         # diccionario con las clases y su frecuencia en el nodo raíz



# Con las descripciones anteriores, ya podemos precisar lo que se pide en eset apartado. 
# Se pide implementar una clase ArbolDecision con el siguiente formato:
  


#  El constructor tiene los siguientes argumentos de entrada:

#     + min_ejemplos_nodo_interior: mínimo número de ejemplos del conjunto de 
#       entrenamiento en un nodo del árbol que se aprende, para que se considere 
#       su división.  
#     + max_prof: profundidad máxima del árbol que se aprende.
#     + n_atrs: número de atributos candidatos a considerar en cada partición
#     + prop_umbral: proporción de ejemplos a considerar cuando se buscan los 
#       umbrales candidatos.    
  
#      

# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente.
#     

# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Método clasifica_prob: recibe UN EJEMPLO y devuelve un diccionario con la predicción
#   de probabilidad de pertenecer a cada clase. Esa probabilidad se calcula como la
#   proporción de ejemplos de clase en la distribución del nodo hoja que da la
#   predicción.

# * Método imprime_arbol: recibe la lista de nombres de cada atributo (columnas) y el
#   nombre del atributo de clasificación, e imprime el árbol de decisión aprendido 
#   (ver ejemplos más abajo) [SUGERENCIA: hacerlo con una función auxiliar recursiva] 


# Si se llama al método de clasificación, o al de impresión, antes de entrenar el modelo,
# se debe devolver (con raise) una excepción:
class ArbolDecisionNoEntrenado(Exception): 
    pass

class ArbolDecision: 
    #Esto sirve para indicar  
    def __init__(self, min_ejemplos_nodo_interior, max_prof, n_atrs, prop_umbral):
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        self.raiz = None

    def entrena(self, X, y):
        
        n_total_atributos = X.shape[1]

        #si n_atrs es None, se consideran todos los atributos como candidatos
        n_atrs = self.n_atrs if self.n_atrs is not None else n_total_atributos

        #Sorteo aleatorio de atributos
        atributos_candidatos=np.random.choice(n_total_atributos,size=n_atrs,replace=False).tolist() 

        self.raiz = CONSTRUYE_ARBOL(X, y, atributos_candidatos, self.min_ejemplos_nodo_interior, self.max_prof, self.prop_umbral)


    def clasifica(self, X):
        if self.raiz is None:
            raise ArbolDecisionNoEntrenado("El clasificador no ha sido entrenado.")
        
        predicciones = []
        for ejemplo in X:
            nodo_actual = self.raiz
            # Recorremos el árbol hasta llegar a una hoja
            while not nodo_actual.es_hoja():
                if ejemplo[nodo_actual.atributo] <= nodo_actual.umbral:
                    nodo_actual = nodo_actual.izq
                else:
                    nodo_actual = nodo_actual.der
            # En nodo_actual ya estamos en una hoja, y la clase que predice se encuentra en nodo_actual.clase
            predicciones.append(nodo_actual.clase)
            return np.array(predicciones)  


    def clasifica_prob(self, ejemplo):
        if self.raiz is None:
            raise ArbolDecisionNoEntrenado("El clasificador no ha sido entrenado.")
        nodo_actual = self.raiz

        # Recorremos el árbol hasta llegar a una hoja
        while not nodo_actual.es_hoja():
            #si el valor del atributo del nodo actual en el ejemplo es menor o igual que el umbral, seguimos por el subárbol izquierdo, 
            # sino seguimos por el subárbol derecho 
            if ejemplo[nodo_actual.atributo]<=nodo_actual.umbral:
                nodo_actual = nodo_actual.izq
            else:
                nodo_actual = nodo_actual.der 

        # En nodo_actual ya estamos en una hoja, y la distribución de clases en esa hoja se encuentra en nodo_actual.distr.
        total_ejemplos = sum(nodo_actual.distr.values())
        
        #devolvemos un diccionario con la predicción de probabilidad de pertenecer a cada clase, calculada como la 
        # proporción de ejemplos de clase en la distribución del nodo hoja que da la predicción.
        return {clase: count / total_ejemplos for clase, count in nodo_actual.distr.items()}

    def imprime_arbol_recursivo(self, nombres_atributos, nombre_clase):
        self ._imprime_arbol_recursivo(self.raiz, nombres_atributos, nombre_clase)
    
    def _imprime_arbol_recursivo(self, nodo, nombres_atributos, nombre_clase, indent=""):
        if nodo.es_hoja():
            print(f"{indent}{nombre_clase}: {nodo.clase} -- {nodo.distr}")
        else:
            print(f"{indent}{nombres_atributos[nodo.atributo]} <= {nodo.umbral:.3f}")
            self._imprime_arbol_recursivo(nodo.izq, nombres_atributos, nombre_clase, indent + "     ")
            print(f"{indent}{nombres_atributos[nodo.atributo]} > {nodo.umbral:.3f}")
            self._imprime_arbol_recursivo(nodo.der, nombres_atributos, nombre_clase, indent + "     ")


# Algunos ejemplos (los resultados pueden variar, debido a la aleatoriedad)
# **************************************************************************

# TITANIC
# -------

# >>> clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
# >>> clf_titanic.entrena(X_train_titanic, y_train_titanic)
# >>> clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Sobrevive")

# Mujer <= 0.000
#      Edad <= 11.000
#           Pclass <= 2.500
#                Sobrevive: 1 -- {1: 10}
#           Pclass > 2.500
#                Sobrevive: 0 -- {0: 13, 1: 8}
#      Edad > 11.000
#           Pclass <= 1.000
#                Sobrevive: 0 -- {0: 62, 1: 30}
#           Pclass > 1.000
#                Sobrevive: 0 -- {0: 270, 1: 32}
# Mujer > 0.000
#      Pclass <= 2.000
#           Edad <= 2.000
#                Sobrevive: 0 -- {0: 1, 1: 1}
#           Edad > 2.000
#                Sobrevive: 1 -- {0: 5, 1: 122}
#      Pclass > 2.000
#           Edad <= 38.500
#                Sobrevive: 1 -- {0: 46, 1: 58}
#           Edad > 38.500
#                Sobrevive: 0 -- {0: 9, 1: 1}

# VOTOS
# -----

# >>> clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
# >>> clf_votos.entrena(Xe_votos, ye_votos)
# >>> nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
# >>> clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")

# Votación 4 <= 0.000
#      Votación 3 <= 0.000
#           Votación 11 <= 0.000
#                Votación 13 <= 0.500
#                     Votación 14 <= -0.500
#                          Partido: democrata -- {'democrata': 2}
#                     Votación 14 > -0.500
#                          Partido: republicano -- {'republicano': 3}
#                Votación 13 > 0.500
#                     Votación 7 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 7 > -1.000
#                          Partido: democrata -- {'democrata': 4}
#           Votación 11 > 0.000
#                Partido: democrata -- {'democrata': 11}
#      Votación 3 > 0.000
#           Partido: democrata -- {'democrata': 149}
# Votación 4 > 0.000
#      Votación 11 <= 0.500
#           Votación 10 <= -1.000
#                Votación 12 <= -1.000
#                     Votación 3 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 3 > -1.000
#                          Partido: republicano -- {'republicano': 2}
#                Votación 12 > -1.000
#                     Votación 3 <= 0.000
#                          Partido: republicano -- {'republicano': 35}
#                     Votación 3 > 0.000
#                          Partido: republicano -- {'democrata': 1, 'republicano': 2}
#           Votación 10 > -1.000
#                Partido: republicano -- {'republicano': 55}
#      Votación 11 > 0.500
#           Votación 7 <= -1.000
#                Votación 3 <= -1.000
#                     Votación 13 <= 0.000
#                          Partido: democrata -- {'democrata': 1}
#                     Votación 13 > 0.000
#                          Partido: republicano -- {'democrata': 2, 'republicano': 9}
#                Votación 3 > -1.000
#                     Partido: democrata -- {'democrata': 6}
#           Votación 7 > -1.000
#                Partido: republicano -- {'republicano': 4}


# IRIS
# ----

    
# >>> clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
# >>> clf_iris.entrena(X_train_iris, y_train_iris)
# >>> clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")



#  Long. Pétalo <= 2.450
#       Clase: 0 -- {0: 33}
#  Long. Pétalo > 2.450
#       Long. Pétalo <= 4.900
#            Anch. Pétalo <= 1.650
#                 Clase: 1 -- {1: 32}
#            Anch. Pétalo > 1.650
#                 Clase: 2 -- {1: 1, 2: 3}
#       Long. Pétalo > 4.900
#            Clase: 2 -- {2: 30}


# CÁNCER DE MAMA
# --------------

# >>> clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer.entrena(Xev_cancer, yev_cancer)

# >>> nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness', 'mean compactness', 'mean concavity',
#        'mean concave points', 'mean symmetry', 'mean fractal dimension',
#        'radius error', 'texture error', 'perimeter error', 'area error',
#        'smoothness error', 'compactness error', 'concavity error',
#        'concave points error', 'symmetry error',
#        'fractal dimension error', 'worst radius', 'worst texture',
#        'worst perimeter', 'worst area', 'worst smoothness',
#        'worst compactness', 'worst concavity', 'worst concave points',
#        'worst symmetry', 'worst fractal dimension']

# >>> clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")


#  mean concave points <= 0.051
#       mean area <= 696.050
#            area error <= 34.405
#                 mean area <= 505.550
#                      Es benigno: 1 -- {1: 172}
#                 mean area > 505.550
#                      worst texture <= 30.145
#                           mean concave points <= 0.050
#                                Es benigno: 1 -- {1: 63}
#                           mean concave points > 0.050
#                                Es benigno: 0 -- {0: 1, 1: 1}
#                      worst texture > 30.145
#                           mean texture <= 24.840
#                                compactness error <= 0.013
#                                     Es benigno: 0 -- {0: 3}
#                                compactness error > 0.013
#                                     Es benigno: 1 -- {1: 2}
#                           mean texture > 24.840
#                                Es benigno: 1 -- {1: 11}
#            area error > 34.405
#                 mean concave points <= 0.032
#                      Es benigno: 1 -- {1: 7}
#                 mean concave points > 0.032
#                      mean perimeter <= 89.175
#                           Es benigno: 0 -- {0: 3}
#                      mean perimeter > 89.175
#                           mean texture <= 20.115
#                                Es benigno: 1 -- {1: 3}
#                           mean texture > 20.115
#                                Es benigno: 0 -- {0: 1}
#       mean area > 696.050
#            mean texture <= 16.190
#                 Es benigno: 1 -- {1: 4}
#            mean texture > 16.190
#                 worst fractal dimension <= 0.066
#                      Es benigno: 1 -- {1: 2}
#                 worst fractal dimension > 0.066
#                      Es benigno: 0 -- {0: 6}
#  mean concave points > 0.051
#       mean area <= 790.850
#            worst texture <= 25.655
#                 mean concave points <= 0.079
#                      mean concave points <= 0.052
#                           Es benigno: 0 -- {0: 1}
#                      mean concave points > 0.052
#                           Es benigno: 1 -- {1: 20}
#                 mean concave points > 0.079
#                      Es benigno: 0 -- {0: 6}
#            worst texture > 25.655
#                 perimeter error <= 1.558
#                      Es benigno: 0 -- {0: 1, 1: 1}
#                 perimeter error > 1.558
#                      Es benigno: 0 -- {0: 37}
#       mean area > 790.850
#            Es benigno: 0 -- {0: 111}



# EJEMPLOS DE RENDIMIENTOS OBTENIDOS CON LOS CLASIFICADORES:
# ----------------------------------------------------------

# Usamos la siguiente función para medir el rendimiento (proporción de aciertos) 
# de un clasificador sobre un conjunto de ejemplos:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/X.shape[0]
    

# Ejemplos (obviamente, el resultado puede variar):


# >>> rendimiento(clf_titanic,X_train_titanic,y_train_titanic)
# 0.8158682634730539
# >>> rendimiento(clf_titanic,X_test_titanic,y_test_titanic)
# 0.7982062780269058

# >>> rendimiento(clf_votos,Xe_votos,ye_votos)
# 0.9827586206896551
# >>> rendimiento(clf_votos,Xp_votos,yp_votos)
# 0.9310344827586207

# >>> rendimiento(clf_iris,X_train_iris,y_train_iris)
#  0.98989898989899
# >>> rendimiento(clf_iris,X_test_iris,y_test_iris)
# 0.9607843137254902

# >>> rendimiento(clf_cancer,Xev_cancer,yev_cancer)
# 0.9956140350877193
# >>> rendimiento(clf_cancer,Xp_cancer,yp_cancer)
# 0.9557522123893806



clf = ArbolDecision(5, 10, 4, 1.0)     
clf.entrena(X_train, y_train)
clf.imprime_arbol_recursivo(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"], "Clase")

# rendimiento
aciertos = sum(clf.clasifica(X_test) == y_test) / len(y_test)
print("Rendimiento test:", aciertos)

# probabilidades de un ejemplo
print("Probabilidades ejemplo 0:", clf.clasifica_prob(X_test[0]))


























# =============================================
# EJERCICIO 3: IMPLEMENTACIÓN DE RANDOM FORESTS
# =============================================

# Usando la clase ArbolDecision, implementar un clasificador Random Forest. 

# Un clasificador Random Forest aplica dos técnicas que reducen el sobreajuste que 
# pudiéramos tener con un único árbol de decisión:

# - En lugar de aprender un árbol. se aprenden varios árboles y a la hora de clasificar
#   nuevos ejemplos, se devuelve la clasificación mayoritaria.
# - Cada uno de esos árboles no se aprende con el conjunto de entrenamiento original, sino
#   con una muestra de ejemplos, obtenido seleccionado los ejemplos aleatoriamente del 
#   conjunto total, CON REEMPLAZO. Además, durante el aprendizaje y en cada nodo, no se usan todos
#   los atributos sino un sunconjunto de ellos obtenidos aleatoriamente (el mismo para todo el árbol). 

# NOTA IMPORTANTE: En la versión estándar del algoritmo Random Forest, el subconjunto de
# atributos a considerar se sortea EN CADA NODO de los árboles que se aprenden. Sin
# embargo, en nuestro caso, como vamos a usar la clase ArbolDecision del ejercicio
# anterior, se va usar el mismo subconjunto de atributos EN CADA ÁRBOL APRENDIDO.

# Concretando, se pide implementar una clase RandomForest con la siguiente estructura:


# class RandomForest:
#     def __init__(self, n_arboles=5,prop_muestras=1.0,
#                        min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
#         .......                   

#     def entrena(self, X, y):
#         .......

#     def clasifica(self, X):
#         .......
    
# Los argumentos del constructor son:

# - n_arboles: el número de árboles que se van a obtener para el clasificador.
# - n_muestras: el número de ejemplos a muestrear para el aprendizaje de cada árbol.
# - El resto de argumentos son los mismos que en el ejercicio anterior, y se usan en el
#   aprendizaje de cada árbol.


# Ejemplos:
# *********

# VOTOS:
# ------

# >>> clf_votos_rf=RandomForest(n_arboles=10,min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=6,prop_umbral=0.8)
# >>> clf_votos_rf.entrena(Xe_votos, ye_votos)
# >>> rendimiento(clf_votos_rf,Xe_votos,ye_votos)
# 0.9517241379310345
# >>> rendimiento(clf_votos_rf,Xp_votos,yp_votos)
# 0.9586206896551724


# >>> clf_cancer_rf = RandomForest(n_arboles=15,min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer_rf.entrena(Xev_cancer, yev_cancer)
# >>> rendimiento(clf_cancer_rf,Xev_cancer,yev_cancer)
# 1.0
# >>> rendimiento(clf_cancer_rf,Xp_cancer,yp_cancer)
# 0.9911504424778761


#------------------------------------------------------------------------------



class RandomForest:
    def __init__(self, n_arboles=5, prop_muestras=1.0, min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
        self.n_arboles = n_arboles
        self.prop_muestras = prop_muestras
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        self.arboles = []

    def entrena(self, X, y):
        self.arboles = []
        n=len(y) 
        #Sacamos el número de atributos del dataset mediante el maximo índice de las columnas de X 
        #y el número de atributos a considerar en cada árbol, que es el mínimo entre n_atrs y el número 
        # total de atributos del dataset
        n_muestras=max(1,int(round(n*self.prop_muestras)))

        #recorremos el número de árboles a aprender, y para cada uno de ellos, obtenemos una muestra aleatoria de ejemplos
        # con reemplazo, y entrenamos un árbol de decisión con esa muestra, usando los mismos valores para los demás hiperparámetros 
        # y el mismo subconjunto de atributos candidatos, que se obtiene aleatoriamente al principio del entrenamiento, y se usa para 
        # todos los árboles.
        for _ in range(self.n_arboles):
            indices_muestra=np.random.choice(n,size=n_muestras,replace=True)  
            x_muestra=X[indices_muestra]
            y_muestra=y[indices_muestra]

            #entrenamos un árbol de decisión con esa muestra, usando los mismos valores para los demás hiperparámetros
            arbol=ArbolDecision(self.min_ejemplos_nodo_interior, self.max_prof, self.n_atrs, self.prop_umbral)
            arbol.entrena(x_muestra, y_muestra)
            self.arboles.append(arbol)


    def clasifica(self, X):
        # Para clasificar un conjunto de ejemplos, obtenemos la clasificación que cada árbol da para esos ejemplos
        votos=np.array([arbol.clasifica(X) for arbol in self.arboles])

        #para cada ejemplo obtenemos la clasificación mayoritaria entre las clasificaciones que dan los árboles
        predicciones=[]
        for j in range(votos.shape[1]):
            clases, counts = np.unique(votos[:, j], return_counts=True)
            predicciones.append(clases[np.argmax(counts)])
        return np.array(predicciones)
    






rf = RandomForest(n_arboles=10, max_prof=3, n_atrs=3)
rf.entrena(X_train, y_train)

aciertos = sum(rf.clasifica(X_test) == y_test) / len(y_test)
print("Rendimiento RF test:", aciertos)
# esperas algo >= que un solo árbol








# =========================================
# EJERCICIO 4: AJUSTANDO LOS CLASIFICADORES
# =========================================

# En este ejercicio vamos a tratar de obtener buenos clasificadores para los 
# los siguientes conjuntos de datos: IMDB, credito, AdultDataset y dígitos.

# ---------------------------
# 4.1 PREPARANDO LOS DATASETS     
# ---------------------------

# Excepto a IMDB, que ya se carga cuando se ejecuta carga_datos.py, el resto 
# tendremos que hacer antes algún preprocesado:
    
# - En X_credito, los atributos son categóricos, así que hay que transformarlos 
#   en numéricos para que se puedan usar con nuestros árboles de decisión. 
#   En el caso de árboles de decisión no es necesario hacer "one hot encoding",
#   sino que basta con codificar los valores de los atributos con números naturales
#   Para ello, SE PIDE USAR el OrdinalEncoder de sklearn.preprocessing (ver manual). 
#   Será necesario también separar en conjunto de prueba y de entrenamiento y
#   validación. 

# - El dataset AdultDataset nos viene es un archivo csv. Cargarlo con 
#   read_csv de pandas, separarlo en entrenamiento y prueba 
#   y aplicarle igualmente OrdinalEncoder, pero sólo a las características desde la 
#   quinta en adelante (ya que las cuatro primeras columnas ya son numéricas). 

# - El dataset de dígitos los podemos obtener a partir de los datos que están en 
#   la carpeta datos/digitdata que se suministra.  Cada imagen viene dada por 28x28
#   píxeles, y cada pixel vendrá representado por un caracter "espacio en
#   blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#   (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#   (es decir, no distinguiremos entre el borde y el interior). En cada
#   conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#   clasificaciones de cada imagen (es decir, el número que representan) vienen
#   en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#   funciones python que lean esos ficheros y obtengan los datos en el mismo
#   formato numpy en el que los necesita el clasificador. 
#   Los datos están ya separados en entrenamiento, validación y prueba. 

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

#   Se pide incluir aquí las definiciones y órdenes necesarias para definir
#   las siguientes variables, con los datasets anteriores como arrays de numpy.


# * X_train_credito, y_train_credito, X_test_credito, y_test_credito
#   conteniendo el dataset de crédito con los atributos numñericos:

# ------------------------------------------------
# Dataset de crédito
# ------------------------------------------------

from datos.credito import *

# Buscamos automáticamente la variable que contiene la lista de datos
datos_credito = None

for nombre in list(globals()):
    valor = globals()[nombre]

    if isinstance(valor, list):
        if len(valor) > 0 and isinstance(valor[0], list):
            if len(valor[0]) == 7:
                datos_credito = valor

if datos_credito is None:
    raise ValueError("No se ha encontrado la lista de datos de crédito")

datos_credito = np.array(datos_credito)

X_credito = datos_credito[:, :6]
y_credito = datos_credito[:, -1]

codificador_credito = OrdinalEncoder()
X_credito = codificador_credito.fit_transform(X_credito)

X_train_credito, X_test_credito, y_train_credito, y_test_credito = particion_entr_prueba(
    X_credito,
    y_credito,
    test=0.25
)








# * X_train_adult, y_train_adult, X_test_adult, y_test_adult
#   conteniendo el AdultDataset con los atributos numéricos:

datos_adult = pd.read_csv("datos/adultDataset.csv")

# Quitamos espacios en blanco en las columnas de texto
datos_adult = datos_adult.replace(r"^\s+|\s+$", "", regex=True)

X_adult = datos_adult.iloc[:, :-1].copy()
y_adult = datos_adult.iloc[:, -1].values

# Las cuatro primeras columnas ya son numéricas
X_adult_numericas = X_adult.iloc[:, :4].values.astype(float)

# Desde la quinta columna en adelante son categóricas
X_adult_categoricas = X_adult.iloc[:, 4:].values

codificador_adult = OrdinalEncoder()
X_adult_categoricas = codificador_adult.fit_transform(X_adult_categoricas)

X_adult = np.concatenate(
    [X_adult_numericas, X_adult_categoricas],
    axis=1
)

X_train_adult, X_test_adult, y_train_adult, y_test_adult = particion_entr_prueba(
    X_adult,
    y_adult,
    test=0.25
)

# Por seguridad, forzamos que las y sean arrays de numpy
y_train_adult = np.array(y_train_adult)
y_test_adult = np.array(y_test_adult)







# * X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg
#   conteniendo el dataset de los dígitos escritos a mano:
    
import os

def leer_imagenes_digitos(nombre_fichero):
    with open(nombre_fichero, "r") as f:
        lineas = f.readlines()

    n_imagenes = len(lineas) // 28
    X = np.zeros((n_imagenes, 28 * 28), dtype=int)

    for i in range(n_imagenes):
        for fila in range(28):
            linea = lineas[i * 28 + fila].rstrip("\n")

            for columna in range(28):
                if columna < len(linea):
                    pixel = linea[columna]

                    if pixel == "+" or pixel == "#":
                        X[i, fila * 28 + columna] = 1
                    else:
                        X[i, fila * 28 + columna] = 0

    return X


def leer_etiquetas_digitos(nombre_fichero):
    with open(nombre_fichero, "r") as f:
        etiquetas = [int(linea.strip()) for linea in f.readlines()]

    return np.array(etiquetas)


ruta_digitos = os.path.join("datos", "digitdata")

X_train_dg = leer_imagenes_digitos(os.path.join(ruta_digitos, "trainingimages"))
y_train_dg = leer_etiquetas_digitos(os.path.join(ruta_digitos, "traininglabels"))

X_valid_dg = leer_imagenes_digitos(os.path.join(ruta_digitos, "validationimages"))
y_valid_dg = leer_etiquetas_digitos(os.path.join(ruta_digitos, "validationlabels"))

X_test_dg = leer_imagenes_digitos(os.path.join(ruta_digitos, "testimages"))
y_test_dg = leer_etiquetas_digitos(os.path.join(ruta_digitos, "testlabels"))

# Alias por si en los ejemplos finales usas estos nombres
X_entr_dg = X_train_dg
y_entr_dg = y_train_dg
X_val_dg = X_valid_dg
y_val_dg = y_valid_dg


# =====================================================
# PRUEBAS PARA COMPROBAR QUE FUNCIONA EL EJERCICIO 4.1
# =====================================================

def comprobar_dataset(nombre, X_train, y_train, X_test, y_test):
    print("\n==============================")
    print("Comprobando dataset:", nombre)
    print("==============================")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    assert isinstance(X_train, np.ndarray), "X_train no es un array de numpy"
    assert isinstance(y_train, np.ndarray), "y_train no es un array de numpy"
    assert isinstance(X_test, np.ndarray), "X_test no es un array de numpy"
    assert isinstance(y_test, np.ndarray), "y_test no es un array de numpy"

    assert X_train.shape[0] == y_train.shape[0], "X_train e y_train no tienen el mismo número de ejemplos"
    assert X_test.shape[0] == y_test.shape[0], "X_test e y_test no tienen el mismo número de ejemplos"

    assert len(X_train.shape) == 2, "X_train no tiene forma de matriz"
    assert len(X_test.shape) == 2, "X_test no tiene forma de matriz"

    assert X_train.shape[1] == X_test.shape[1], "Train y test no tienen el mismo número de atributos"

    assert np.issubdtype(X_train.dtype, np.number), "X_train no es numérico"
    assert np.issubdtype(X_test.dtype, np.number), "X_test no es numérico"

    assert not np.isnan(X_train).any(), "Hay valores NaN en X_train"
    assert not np.isnan(X_test).any(), "Hay valores NaN en X_test"

    print("Clases train:", np.unique(y_train, return_counts=True))
    print("Clases test:", np.unique(y_test, return_counts=True))

    print("Primer ejemplo de X_train:")
    print(X_train[0])

    print("OK:", nombre, "cargado correctamente")


# Crédito
comprobar_dataset(
    "Crédito",
    X_train_credito,
    y_train_credito,
    X_test_credito,
    y_test_credito
)


# Adult
comprobar_dataset(
    "Adult",
    X_train_adult,
    y_train_adult,
    X_test_adult,
    y_test_adult
)


# Dígitos
print("\n==============================")
print("Comprobando dataset: Dígitos")
print("==============================")

print("X_train_dg:", X_train_dg.shape)
print("y_train_dg:", y_train_dg.shape)
print("X_valid_dg:", X_valid_dg.shape)
print("y_valid_dg:", y_valid_dg.shape)
print("X_test_dg:", X_test_dg.shape)
print("y_test_dg:", y_test_dg.shape)

assert isinstance(X_train_dg, np.ndarray), "X_train_dg no es un array de numpy"
assert isinstance(y_train_dg, np.ndarray), "y_train_dg no es un array de numpy"
assert isinstance(X_valid_dg, np.ndarray), "X_valid_dg no es un array de numpy"
assert isinstance(y_valid_dg, np.ndarray), "y_valid_dg no es un array de numpy"
assert isinstance(X_test_dg, np.ndarray), "X_test_dg no es un array de numpy"
assert isinstance(y_test_dg, np.ndarray), "y_test_dg no es un array de numpy"

assert X_train_dg.shape[0] == y_train_dg.shape[0], "Train de dígitos mal emparejado"
assert X_valid_dg.shape[0] == y_valid_dg.shape[0], "Validación de dígitos mal emparejada"
assert X_test_dg.shape[0] == y_test_dg.shape[0], "Test de dígitos mal emparejado"

assert X_train_dg.shape[1] == 784, "Las imágenes de entrenamiento no tienen 784 píxeles"
assert X_valid_dg.shape[1] == 784, "Las imágenes de validación no tienen 784 píxeles"
assert X_test_dg.shape[1] == 784, "Las imágenes de test no tienen 784 píxeles"

assert set(np.unique(X_train_dg)).issubset({0, 1}), "X_train_dg debe contener solo 0 y 1"
assert set(np.unique(X_valid_dg)).issubset({0, 1}), "X_valid_dg debe contener solo 0 y 1"
assert set(np.unique(X_test_dg)).issubset({0, 1}), "X_test_dg debe contener solo 0 y 1"

print("Clases train dígitos:", np.unique(y_train_dg, return_counts=True))
print("Clases valid dígitos:", np.unique(y_valid_dg, return_counts=True))
print("Clases test dígitos:", np.unique(y_test_dg, return_counts=True))

print("Primer dígito vectorizado:")
print(X_train_dg[0])

print("OK: Dígitos cargado correctamente")


# IMDB ya venía cargado con carga_datos.py
print("\n==============================")
print("Comprobando dataset: IMDB")
print("==============================")

print("X_train_imdb:", X_train_imdb.shape)
print("y_train_imdb:", y_train_imdb.shape)
print("X_test_imdb:", X_test_imdb.shape)
print("y_test_imdb:", y_test_imdb.shape)

assert X_train_imdb.shape[0] == y_train_imdb.shape[0], "Train IMDB mal emparejado"
assert X_test_imdb.shape[0] == y_test_imdb.shape[0], "Test IMDB mal emparejado"
assert np.issubdtype(X_train_imdb.dtype, np.number), "X_train_imdb no es numérico"
assert np.issubdtype(X_test_imdb.dtype, np.number), "X_test_imdb no es numérico"

print("Clases train IMDB:", np.unique(y_train_imdb, return_counts=True))
print("Clases test IMDB:", np.unique(y_test_imdb, return_counts=True))

print("OK: IMDB cargado correctamente")

print("\nTODAS LAS PRUEBAS DEL EJERCICIO 4.1 HAN PASADO CORRECTAMENTE")


# -----------------------------
# 4.2 AJUSTE DE HIPERPARÁMETROS     
# -----------------------------

# En nuestra implementación de RandomForest tenemos los siguientes 
# hiperparámetros: 

# n_arboles
# prop_muestras
# min_ejemplos_nodo_interior
# max_prof
# n_atrs
# prop_umbral

# Se trata ahora de encontrar, en cada dataset, una buena combinación de valores para esos 
# hiperparámetros, tratando de obtener un buen rendimiento de los clasificadores. Hacerlo
# usando un conjunto de validación: según se ha visto en la teoría, esto consiste en particionar  
# en entrenamiento, validación y prueba, entrenando por cada combinación de hiperparámetros 
# con el conjunto de entrenamiento y evaluando el rendimiento en validación. El entrenamiento final 
# con la mejor combinación ha de hacerse en la unión de entrenamiento y validación.
    

# NO ES NECESARIO ser demasiado exhaustivo, basta con probar algunas combinaciones, 
# pero sí es importante describir el proceso realizado y las mejores combinaciones 
# encontradas en cada caso. 
# DEJAR ESTE APARTADO COMENTADO, para que no se ejecuten las pruebas realizadas cuando se cargue
# el archivo. 

# ----------------------------












# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************

# EJEMPLOS DE PRUEBA

# LAS SIGUIENTES LLAMADAS SERÁN EJECUTADAS POR EL PROFESOR EL DÍA DE LA PRESENTACIÓN.
# UNA VEZ IMPLEMENTADAS LAS DEFINICIONES Y FUNCIONES NECESARIAS
# Y REALIZADOS LOS AJUSTES DE HIPERPARÁMETROS, 
# DEJAR COMENTADA CUALQUIER LLAMADA A LAS FUNCIONES QUE SE TENGA EN ESTE ARCHIVO 
# Y DESCOMENTAR LAS QUE VIENEN A CONTINUACIÓN.
# UNA VEZ HECHO ESO, ENTREGAR TAMBIÉN UN ARCHIVO trabajo_aia_25_26_parte_I_sc.py
# SIN COMENTARIOS AL CÓDIGO. 


# EN EL APARTADO FINAL DE "RENDIMIENTOS FINALES RANDOM FOREST", USAR LA MEJOR COMBINACIÓN DE 
# HIPERPARÁMETROS QUE SE HAYA OBTENIDO EN CADA CASO, EN LA FASE DE AJUSTE DEL EJERCICIO 4

# El ARCHIVO trabajo_aia_23_24_parte_I_sc.py SERÁ CARGADO POR EL PROFESOR, 
# TENIENDO EN LA MISMA CARPETA LOS ARCHIVOS OBTENIDOS
# DESCOMPRIMIENDO datos_trabajo_aia.zip.
# ES IMPORTANTE QUE LO QUE SE ENTREGA SE PUEDA CARGAR SIN ERRORES Y QUE SE EJECUTEN LOS 
# EJEMPLOS QUE VIENEN A CONTINUACIÓN. SI ALGUNO DE LOS EJERCICIOS NO SE HA REALIZADO 
# O DEVUELVE ALGÚN ERROR, DEJAR COMENTADOS LOS CORRESPONDIENTES EJEMPLOS, 
# PARA EViTAR LOS ERRORES EN LA CARGA Y EJECUCIÓN.   



# *********** DESCOMENTAR A PARTIR DE AQUÍ

# print("************ PRUEBAS EJERCICIO 1:")
# print("**********************************\n")
# Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
# print("Partición votos: ",y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
# print("Proporción original en votos: ",np.unique(y_votos,return_counts=True))
# print("Estratificación entrenamiento en votos: ",np.unique(ye_votos,return_counts=True))
# print("Estratificación prueba en votos: ",np.unique(yp_votos,return_counts=True))
# print("\n")

# Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
# print("Proporción original en cáncer: ", np.unique(y_cancer,return_counts=True))
# print("Estratificación entr-val en cáncer: ",np.unique(yev_cancer,return_counts=True))
# print("Estratificación prueba en cáncer: ",np.unique(yp_cancer,return_counts=True))
# Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
# print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer,return_counts=True))
# print("Estratificación validación cáncer: ",np.unique(yv_cancer,return_counts=True))
# print("\n")

# Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
# print("Estratificación entrenamiento crédito: ",np.unique(ye_credito,return_counts=True))
# print("Estratificación prueba crédito: ",np.unique(yp_credito,return_counts=True))
# print("\n\n\n")





# print("************ PRUEBAS EJERCICIO 2:")
# print("**********************************\n")

# clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
# clf_titanic.entrena(X_train_titanic, y_train_titanic)
# clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Partido")
# rend_train_titanic = rendimiento(clf_titanic,X_train_titanic,y_train_titanic)
# rend_test_titanic = rendimiento(clf_titanic,X_test_titanic,y_test_titanic)
# print(f"****** Rendimiento DT titanic train: {rend_train_titanic}")
# print(f"****** Rendimiento DT titanic test: {rend_test_titanic}\n\n\n\n ")




# clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
# clf_votos.entrena(Xe_votos, ye_votos)
# nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
# clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")
# rend_train_votos = rendimiento(clf_votos,Xe_votos,ye_votos)
# rend_test_votos = rendimiento(clf_votos,Xp_votos,yp_votos)
# print(f"****** Rendimiento DT votos en train: {rend_train_votos}")
# print(f"****** Rendimiento DT votos en test:  {rend_test_votos}\n\n\n\n")



# clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
# clf_iris.entrena(X_train_iris, y_train_iris)
# clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")
# rend_train_iris = rendimiento(clf_iris,X_train_iris,y_train_iris)
# rend_test_iris = rendimiento(clf_iris,X_test_iris,y_test_iris)
# print(f"********************* Rendimiento DT iris train: {rend_train_iris}")
# print(f"********************* Rendimiento DT iris test: {rend_test_iris}\n\n\n\n ")





# clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# clf_cancer.entrena(Xev_cancer, yev_cancer)
# nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#         'mean smoothness', 'mean compactness', 'mean concavity',
#         'mean concave points', 'mean symmetry', 'mean fractal dimension',
#         'radius error', 'texture error', 'perimeter error', 'area error',
#         'smoothness error', 'compactness error', 'concavity error',
#         'concave points error', 'symmetry error',
#         'fractal dimension error', 'worst radius', 'worst texture',
#         'worst perimeter', 'worst area', 'worst smoothness',
#         'worst compactness', 'worst concavity', 'worst concave points',
#         'worst symmetry', 'worst fractal dimension']
# clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")
# rend_train_cancer = rendimiento(clf_cancer,Xev_cancer,yev_cancer)
# rend_test_cancer = rendimiento(clf_cancer,Xp_cancer,yp_cancer)
# print(f"***** Rendimiento DT cancer en train: {rend_train_cancer}")
# print(f"***** Rendimiento DT cancer en test: {rend_test_cancer}\n\n\n")



# print("************ RENDIMIENTOS FINALES RANDOM FOREST")
# print("************************************************\n")


# # ATENCIÓN: EN CADA CASO, INCORPORAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS 
# # QUE SE HA OBTENIDO EN EL PROCESO DE AJUSTE



# print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE IMDB:")
# RF_IMDB=RandomForest(?????????????????) # ATENCIÓN: incorporar aquí los mejores valoeres de los parámetros tras el ajuste
# RF_IMDB.entrena(X_train_imdb,y_train_imdb) 
# print("Rendimiento RF entrenamiento sobre imdb: ",rendimiento(RF_IMDB,X_train_imdb,y_train_imdb))
# print("Rendimiento RF test sobre imdb: ",rendimiento(RF_IMDB,X_test_imdb,y_test_imdb))
# print("\n")




# print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE CRÉDITO:")

# RF_CREDITO=RandomForest(??????????????) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
# RF_CREDITO.entrena(X_train_credito,y_train_credito) 
# print("Rendimiento RF entrenamiento sobre crédito: ",rendimiento(RF_CREDITO,X_train_credito,y_train_credito))
# print("Rendimiento RF  test sobre crédito: ",rendimiento(RF_CREDITO,X_test_credito,y_test_credito))
# print("\n")


# print("==== MEJOR RENDIMIENTO RF SOBRE ADULT:")

# RF_ADULT=RandomForest(??????????????) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
# RF_ADULT.entrena(X_train_adult,y_train_adult) 
# print("Rendimiento RF  entrenamiento sobre adult: ",rendimiento(RF_ADULT,X_train_adult,y_train_adult))
# print("Rendimiento RF  test sobre adult: ",rendimiento(RF_ADULT,X_test_adult,y_test_adult))
# print("\n")


# print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
# RF_DG=RandomForest(?????????????) # ATENCIÓN: incorporar aquí los mejores valors de losparámetros tras el ajuste
# RF_DG.entrena(X_entr_dg,y_entr_dg)
# print("Rendimiento RF entrenamiento sobre dígitos: ",rendimiento(RF_DG,X_entr_dg,y_entr_dg))
# print("Rendimiento RF validación sobre dígitos: ",rendimiento(RF_DG,X_val_dg,y_val_dg))
# print("Rendimiento RF test sobre dígitos: ",rendimiento(RF_DG,X_test_dg,y_test_dg))








