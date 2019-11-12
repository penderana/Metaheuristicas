#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:37:40 2019

@author: Carlos
"""
from scipy.io import arff
import pandas as pd
import numpy as np 
import random as r
from sklearn.neighbors import KDTree
from time import time
from sklearn.utils import shuffle
import math as m
import matplotlib.pyplot as plt



def comprobar(pesos,i,boolean=True):
    _pesos = np.copy(pesos)
    _pesos[_pesos < 0.2 ] = 0
    
    train,test = carga_datos(data,i)
    x_train,y_train = separar(train)
    x_test,y_test = separar(test)
    
    x_train = (x_train*_pesos)[:,_pesos > 0.2]
    x_test = (x_test*_pesos)[:,_pesos > 0.2]

    tree = KDTree(x_train, leaf_size=1)
    dis,vecinos = tree.query(x_test,k=1)
    vecinos = vecinos[:,0]
#    print(y_train,y_test)
    aciertos = np.mean( y_train[vecinos] == y_test) * 100
#    for m in range(0,len(datos_test)):
#
#        datos_test_ = datos_test[m]
#        dis,ind = tree.query(datos_test_.reshape(1,-1), k=1)
#
#        ind = ind[0][0]
#
#        if etiquetas_train[ind] == etiquetas_test[m]:
#            aciertos += 1
     

    calif = califica_pesos(pesos)
#    aciertos = aciertos * 100 / len(datos_test)
    calif = calif *100 / len(pesos)
    
    if boolean:
      print("CONJUNTO DE DATOS ",i,": ","%_clas: ",aciertos,"%red: ",calif)
    return (aciertos + calif) /2


def multiplica_pesos(x,pesos):
    
    M = len(x[0]) - 1
    data = x[:,0:M]
    for i in range(0,len(x)):
        for j in range(0,len(x[0])-1):
            if pesos[j] >= 0.2:
                data[i][j] = x[i][j]*pesos[j]
            else:
                data[i][j] = 0

    return data

#truncamos los pesos  
def corregir_pesos(pesos):
    w = pesos
    maximo = max(pesos)

    for i in range(0,len(pesos)):
        if pesos[i] < 0:
            w[i] = 0
        elif pesos[i] > 1:
            w[i] = (pesos[i]/maximo)

    return w
#funcion que cuenta cuantos pesos tienen valor inferior a 0.2
def califica_pesos(pesos):
    return len([p for p in pesos if p < 0.2])

def separar(data):
    datos = []  
    etiquetas = []
    
    for data1 in data:
        etiquetas.append(data1[-1])
        datos.append(data1[0:-1])
        
    etiquetas = np.array(etiquetas, np.float64)  #Conversion al tipo correcto
    datos = np.array(datos, np.float64)
    
    return datos,etiquetas

def loaddata(path):
    f = path
    data, meta = arff.loadarff(f)
    df_data = pd.DataFrame(data)
    data = df_data.values
    
    try:
        float(data[0][len(data[0])-1])
    except:
        variables = []
        for x in data:
            if x[-1] not in variables:
                variables.append(x[len(x)-1])
                
        numeros = list(range(0,len(variables)))
        diccionario = {}
        for i in range(0,len(numeros)):
            diccionario.update({variables[i]:numeros[i]})
        
        for i in range(0,len(data)):
            data[i][len(data[0])-1] = diccionario.get(data[i][len(data[0])-1])
            
        print("Etiquetas modificadas de la siguiente forma: ",diccionario)
    print("Cargado ", path)
    
    return data, meta

#pesos aleatorios
def inicia_pesos(M):
    np.random.seed(1)

    w = []
    
    for i in range(0,M):
        numero = r.randrange(100)
        w.append(numero/100)
        
    return w


def cruce_aritmetico(padre,madre,iteracion):
    M = len(padre.w)
    array_padre = padre.w
    array_madre = madre.w
    
    alfas = np.random.rand(M,1)
    hijo1 = ( 1 - alfas ) * array_padre + alfas * array_madre
    hijo2 = alfas *array_padre + ( 1 - alfas ) * array_madre
    

#    hijo = cromosoma(M,iteracion,hijo1[0])
#    hermano = cromosoma(M,iteracion,hijo2[0])
    
    
    return hijo1[0],hijo2[0]

def cruce_blx(padre,madre,alfa,iteracion):
    
    M = len(padre.w)
    
    hijo1 = []
    hijo2 = []
    
    for i in range(0,M):
        Cmax = max(padre.w[i],madre.w[i])
        Cmin = min(padre.w[i],madre.w[i])
        I = Cmax - Cmin
        cota_inferior = (Cmin-I*alfa)
        cota_superior = (Cmax-I*alfa)
#        print(cota_inferior,cota_superior)
        hijo1.append(r.uniform(cota_inferior,cota_superior))
        hijo2.append(r.uniform(cota_inferior,cota_superior))
        
#    hijo = cromosoma(M,iteracion,hijo1)
#    hermano = cromosoma(M,iteracion,hijo2)
    
    
    return hijo1,hijo2

def carga_datos(data,iteracion):
    i = iteracion
    N = len(data)
    tam = N // 5
    if i == 0:
        x = data[tam:N]
        y = data[0:tam]
    else:
        tope = (i+1)*tam
        if tope > N:
            tope = N
        quitar = range(i*tam,tope)
            
        x = np.delete(data,quitar,0)
        y = data[i*tam:tope]
    
    return x,y
 
def comprobar_(pesos,iteracion):
#def comprobar_(pesos,iteracion):
    _pesos = np.copy(pesos)
    _pesos[_pesos < 0.2] = 0
    
    train,test = carga_datos(data,iteracion) #no hacemos nada con la y.
    x_train,y_train = separar(train)

    x_train = (x_train*_pesos)[:,_pesos > 0.2]
    
    try:
        tree = KDTree(x_train)
        dis,vecinos = tree.query(x_train,k=2)
        vecinos = vecinos[:,1]
        aciertos = np.mean( y_train[vecinos] == y_train)*100
    except:
        aciertos = 0
        
    calif = califica_pesos(pesos) * 100 / len(pesos)
  

    return (aciertos + calif) /2

def busqueda_local(j,pesos_): # REDUCIDAS LAS EVALUACIONES. 15000 -> 1000
    #solucion inicial:
    if len(pesos_) == 0:
        pesos = inicia_pesos(M)
    else:
        pesos = np.copy(pesos_)
#    pesos = greedy()
#    pesos = np.zeros(M,np.float64)

    desviacion = 0.3
    O = len(pesos)
    calidad = comprobar_(pesos,j)
    iters = 1
    no_mejora = 0
    while iters < 1000 and no_mejora < 20*O :
        for i in range(0,O):
            prev = pesos[i]
            valor = np.random.normal(0,desviacion)
            pesos[i] = pesos[i] + valor
            if pesos[i] < 0:
                pesos[i] = 0
            elif pesos[i] > 1:
                maxi = max(pesos)
                pesos[i] = maxi / pesos[i]
            calidad1 = comprobar_(pesos,j)
            iters += 1
            
            if calidad1 > calidad:
#                pesos = copia_pesos
                no_mejora = 0
                calidad = calidad1
                
                break
            else:
                pesos[i] = prev
                no_mejora += 1
            
        
    return pesos


class cromosoma:
    def __init__(self,tam,iteracion,cromosoma):
        if len(cromosoma) == 0:
            self.w = inicia_pesos(tam)
        else:
            self.w = cromosoma
        self.calificacion = comprobar_(self.w,iteracion)
        
    def comprobar(self,iteracion):
        self.calificacion = comprobar_(self.w,iteracion)
        
def mutacion(pesos,des_tipica,pg):
    p = np.copy(pesos)
    for i in range(0,len(pesos)):
        aleatorio = np.random.rand(1,1)
        if aleatorio < pg:
            p[i] += np.random.normal(0.0, des_tipica)
            
    return corregir_pesos(p)

def ils(k):
    pesos = inicia_pesos(M)
    pesos_bl = busqueda_local(k,pesos)
      
    calidad_uno = comprobar_(pesos,k)
    calidad_dos = comprobar_(pesos_bl,k)
    brusco = []
    suave = []
    
    if calidad_uno > calidad_dos:
        mejor = pesos
        calidad_mejor = calidad_uno
    else:
        mejor = pesos_bl
        calidad_mejor = calidad_dos

    orig = np.copy(mejor)
    mejor_ = mutacion_suave(mejor)
    mejor = mutacion(mejor,0.4,0.1)

    brusco.append(np.linalg.norm(np.array(orig)-np.array(mejor)))
    suave.append(np.linalg.norm(np.array(orig)-np.array(mejor_)))
    for i in range(0,14):
        mejor = np.copy(mejor)
        pesos = busqueda_local(k,mejor)
        calidad_mutado = comprobar_(pesos,k)
        
        if calidad_mutado > calidad_mejor:
            calidad_mejor = calidad_mutado
            suave.append(np.linalg.norm(np.array(pesos)-np.array(mutacion_suave(pesos))))
            mejor = mutacion(pesos,0.4,0.1)
            brusco.append(np.linalg.norm(np.array(pesos)-np.array(mejor)))

        #ELSE: NO MODIFICACAMOS NADA
        
    return mejor,suave,brusco

  
def mutacion_suave(pesos):
    posicion = r.randrange(M)
    pesos[posicion] += np.random.normal(0.0, 0.4)
    return corregir_pesos(pesos)

def enfriamiento_simulado(k):
    pesos = inicia_pesos(M)
    mejor = pesos
    mu = 0.3
    phi = 0.3
    calidad_antes = comprobar_(pesos,k)
    T_cero = calidad_antes*mu/(-m.log(phi))
#    T_cero = mu/(-m.log(phi))
    T_f = 10**(-3)
    T = T_cero
    N = len(datos)
    max_vecinos = 10 * N 
    max_exitos = 0.1 * max_vecinos
    max_evas = 15000
    max_enfriamientos = (max_evas / max_vecinos)
    desviacion = 0.3
    vecinos = 0
    exitos = 0
    calidad_mejor = 0
    
    temperaturas = []
    temperaturas.append(T_cero)
    iterazione = []
    iterazione.append(0)
    
    B = (T_cero - T_f)/max_enfriamientos*T_cero*T_f
    iters = 0
    while T > T_f and iters < max_enfriamientos:
#        print(T,T_f)
        vecinos = 0
        exitos = 0  
        while vecinos < max_vecinos and exitos < max_exitos:
            posicion = r.randrange(M)
            pesos_mutados = np.copy(pesos)

            pesos_mutados[posicion] += np.random.normal(0,desviacion)

            pesos_mutados = corregir_pesos(pesos_mutados)
            vecinos += 1
            
#            calidad_antes = comprobar_(pesos,k)
            calidad_despues = comprobar_(pesos_mutados,k)
            
            
            delta = calidad_despues - calidad_antes
            
            
            if delta > 0:
                calidad_antes = calidad_despues
                pesos = pesos_mutados
                exitos += 1
                if calidad_antes > calidad_mejor:
                    calidad_mejor = calidad_despues
                    mejor = pesos
                    
            else:
                prob = np.exp(delta/T)
    
                alea = np.random.rand(1,1)
                
                if alea >= prob:
                    calidad_antes = calidad_despues
                    pesos = pesos_mutados
                    exitos += 1
                    
        T = T / (1 + B * T)
#        T = 0.95* T    
        iters += 1
        iterazione.append(iters)
        temperaturas.append(T)
    
    return mejor,temperaturas,iterazione
            
    
def evolucion_diferencial(k,boolean):
    
# =============================================================================
#     DOS TIPOS DE EVOLUCION DIFERENCIAL: RAND Y CURRENT-TO-BEST
#     boolean -> rand
#     !boolean -> current-to-best
# =============================================================================
    
    N = 50 # TENEMOS 50 CROMOSOMAS.
    p_cr = 0.5
    F = 0.5
    evaluazione = []
    funzione = []
    poblacion = []
    for i in range(0,N): #GENERAMOS LA POBLACION INICIAL
        poblacion.append(cromosoma(M,k,[]))
    poblacion = np.array(poblacion, cromosoma)
    evas = N

    while evas < 15000:
        poblacion_off = []

        for i in range(0,N):
            
            oofspring_w = []
            
            if boolean:
                parent1_pos = r.randrange(N)
                parent2_pos = r.randrange(N)
                while parent2_pos == parent1_pos:
                    parent2_pos = r.randrange(N)
                parent3_pos = r.randrange(N)
                while parent3_pos == parent2_pos or parent3_pos == parent1_pos:
                    parent3_pos = r.randrange(N)
                    
               
                parent1 = poblacion[parent1_pos].w
                parent2 = poblacion[parent2_pos].w
                parent3 = poblacion[parent3_pos].w
                
                mej = 0
                for j in range(1,N):
                    if poblacion[j].calificacion > poblacion[mej].calificacion:
                        mej = j
                evaluazione.append(evas)
                funzione.append(poblacion[mej].calificacion)       
                for j in range(0,M):
                    if np.random.rand(1,1) <= p_cr:
                        
                        oofspring_w.append(parent1[j]+F*(parent2[j]-parent3[j]))

                    else:
                        oofspring_w.append(poblacion[i].w[j])
                
            else:
                parent1_pos = r.randrange(N)
                parent2_pos = r.randrange(N)
                while parent2_pos == parent1_pos:
                    parent2_pos = r.randrange(N)
                
            
                parent1 = poblacion[parent1_pos].w
                parent2 = poblacion[parent2_pos].w
                
                
                mej = 0
                for j in range(1,N):
                    if poblacion[j].calificacion > poblacion[mej].calificacion:
                        mej = j

                
#                parent1 = mutacion(parent1,0.4,0.1)
#                parent2 = mutacion(parent2,0.4,0.1)
                
                evaluazione.append(evas)
                funzione.append(poblacion[mej].calificacion)
                for j in range(0,M):
                    if np.random.rand(1,1) <= p_cr:
                        oofspring_w.append(poblacion[i].w[j]+F*(poblacion[mej].w[j]-poblacion[i].w[j])+F*(parent1[j]-parent2[j]))
                    else:
                        oofspring_w.append(poblacion[i].w[j])

               
            oofspring_w = corregir_pesos(oofspring_w)
            
            poblacion_off.append(cromosoma(M,k,oofspring_w))
            
            evas += 1
            
        poblacion_off = np.array(poblacion_off, cromosoma)

        poblacion_f = []

        for j in range(0,N):
            if poblacion[j].calificacion > poblacion_off[j].calificacion:
                poblacion_f.append(poblacion[j])
            else:
                poblacion_f.append(poblacion_off[j])
                
        poblacion_f = np.array(poblacion_f, cromosoma)
        poblacion = poblacion_f
        
    mej = 0
    for j in range(1,N):
        if poblacion[j].calificacion > poblacion[mej].calificacion:
            mej = j
            
    return poblacion[mej].w,evaluazione,funzione
    
archivos = ['datos/colposcopy.arff','datos/ionosphere.arff','datos/texture.arff']
archivo =  'datos/colposcopy.arff'

mejor_resultado = 0


vara = time()
for archivo in archivos:
    data, meta = loaddata(archivo)
    data = shuffle(data)
    datos,etiquetas = separar(data)
    M = len(datos[0])

    for k in range(0,5):
        var = time()
        kek,uno,dos = ils(k)
        resultado = comprobar(kek,k)
       
        print("CALIDAD ILS:",resultado)
        print("TIEMPO:",time()-var)
        mejor,temperaturas,iterazione = enfriamiento_simulado(k)
        resultado = comprobar(busqueda_local(k,[]),k)
        
        print("CALIDAD ES:",resultado)
        print("TIEMPO:",time()-var)
        if resultado > mejor_resultado:
                mejor_resultado = resultado
                mejor_archivo = archivo
                mejor_k = k
        booleanos = [True,False]
        for boolean in booleanos:
            var = time()
            kek,uno,dos = evolucion_diferencial(k,boolean)
            resultado = comprobar(kek,k)
            
            print("CALIDAD DE-",boolean,":",resultado)
        
            print("TIEMPO:",time()-var)
        
            if resultado > mejor_resultado:
                mejor_resultado = resultado
                mejor_archivo = archivo
                mejor_k = k
        
print("-------------------------------\n")
print("MEJOR RESULTADO:",round(mejor_resultado,2),"EN:",round(time()-vara,2),"SEGUNDOS.")
print("EN LA PARTICION",mejor_k,"DEL ARCHIVO",mejor_archivo)
    
data, meta = loaddata('datos/colposcopy.arff')
data = shuffle(data)
datos,etiquetas = separar(data)
M = len(datos[0])
resultado,temperaturas,iters = enfriamiento_simulado(0)

plt.plot(iters,temperaturas)
plt.xlabel("iteraciones")
plt.ylabel("Valor de la temperatura")
plt.title("Descenso de la temperatura")
plt.show()

data, meta = loaddata('datos/colposcopy.arff')
data = shuffle(data)
datos,etiquetas = separar(data)
M = len(datos[0])
resultado,temperaturas,iters = evolucion_diferencial(0,False)
resultado,temperaturas2,iters2 = evolucion_diferencial(0,True)
plt.plot(temperaturas,iters,label="current-to-best")
plt.plot(temperaturas2,iters2,label="rand",color="red")
plt.xlabel("evaluaciones")
plt.legend()
plt.ylabel("funcion objetivo")
plt.title("Convergencia")
plt.show()
