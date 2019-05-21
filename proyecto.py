# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy 
##usamos la semilla 100 para hacer nuestro proyecto producible
numpy.random.seed(100)

#Grafico de codo
def elbowPlot(data,maxKClusters):
    inertias=list()
    for i in range(1,maxKClusters+1):
        myCluster=KMeans(n_clusters=i)
        myCluster.fit(data)   
        inertias.append(myCluster.inertia_)  
    plt.figure() 
    x=[i for i in range(1,maxKClusters+1)]
    y=[i for i in inertias]
    plt.plot(x,y, 'ro-', markersize=8, lw=2)
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

######---------------------MAIN-------------------------------------
df = pd.read_csv("newdatos.csv",nrows=20000, encoding = "ISO-8859-1")
list(df)


#eliminamos la columna de departamentos 
df=df.drop(['ESTU_DEPTO_RESIDE'],axis=1)
#graficamos el elbow plot
elbowPlot(df,10)
#creamos los clusters con 2 centroides
cluster=KMeans(n_clusters=2)
cluster.fit(df)
#guardamos los centroides en una variable, se convierte en df, y se agregan los nombres de columnas
centros=cluster.cluster_centers_
centros=pd.DataFrame(centros)
centros.columns=list(df)

#exportamos los centroides a un nuevo archivo csv
#centros.to_csv(r'centros.csv')


#y asignamos los cluster a los que pertenece cada dato como una nueva columna
df['cluster']=cluster.labels_

#grafica de puntajes promedio y rango de puntajes por cada cluster
df.plot.scatter(x='cluster',y='PUNT_GLOBAL',c='cluster',colormap='viridis')


#exportamos los datos de los 20 mil estudiantes con la columna de cluster agregadas
#a estos datos 
#df.to_csv(r'icfesConCluster.csv')


