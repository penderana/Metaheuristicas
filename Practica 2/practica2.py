# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:46:34 2019

@author: Carlos Rana
"""

#PRACTICA 2 MH

from scipy.io import arff
import numpy as np 
import random as r
from time import time
import pandas as pd
from sklearn.neighbors import KDTree
import operator
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

np.random.seed(0)

#FUNCION DEL KNN
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


class cromosoma:
    def __init__(self,tam,iteracion,cromosoma):
        if len(cromosoma) == 0:
            self.w = inicia_pesos(tam)
        else:
            self.w = cromosoma
        self.calificacion = comprobar_(self.w,iteracion)
        
    def comprobar(self,iteracion):
        self.calificacion = comprobar_(self.w,iteracion)
        
def busqueda_local(j,pesos_):
    #solucion inicial:
    if len(pesos_) == 0:
        pesos = inicia_pesos(M)
    else:
        pesos = pesos_
#    pesos = greedy()
#    pesos = np.zeros(M,np.float64)

    desviacion = 0.3
    O = len(pesos)
    calidad = comprobar_(pesos,j)
    iters = 1
    no_mejora = 0
    while iters < 15000 and no_mejora < 20*O :
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

def genetico_generacional(data,iteracion,boolean):
    datos,etiquetas = separar(data)
    poblacion = []
# =============================================================================
#     AG GENERACIONAL:
#         BOOLEAN = CRUCE BLX ALFA
#         !BOOLEAN = CRUCE ARITMETICO PONDERADO
# =============================================================================
    N = 30 #TENEMOS 30 CROMOSOMAS
    M = len(datos[0])
#    calificaciones = []
    vacio = []
    for i in range(0,N): #GENERAMOS LA POBLACION INICIAL
        poblacion.append(cromosoma(M,iteracion,vacio))
#        calificaciones.append(poblacion[i].calificacion)
    poblacion = np.array(poblacion, cromosoma)
    
    Pm = 0.001
    Pc = 0.7
    num_cruces = Pc * N / 2
    num_cruces = int(num_cruces)
    i = N
    mutaciones = (int)(N * M * Pm)
    
    while i < 15000: #evaluaciones
#        antes = round(i/150,2)
#        print(antes,"%")
#        var = time()
        
# =============================================================================
#         PARTE 0: GUARDAMOS EL MEJOR CROMOSOMA POR SI LO PERDEMOS
# =============================================================================
        mex = 0
        
        for k in range(1,N):
#            print(poblacion[k].calificacion)
            if poblacion[k].calificacion > poblacion[mex].calificacion:
                mex = k
                
        mejor_individuo = poblacion[mex]
#        print(mejor_individuo.calificacion)
#        var2 = time()
#        print("timpo parte 0:",var2-var)
# =============================================================================
#         PARTE 1: TORNEO BINARIO. ELEGIMOS DOS CROMOSOMAS AL AZAR Y
#         LO ANIADIMOS A LA NUEVA POBLACION.
# =============================================================================
        
        poblacion_prima = []
#        aleatorios = r.sample(range(N),k=N)
        
        for j in range(0,N): 
            w1 = r.randrange(N)
            w2 = r.randrange(N)
#            w1 = aleatorios[j]
#            w2 = aleatorios[N-j-1] 
            
            if poblacion[w1].calificacion > poblacion[w2].calificacion:
#                poblacion_prima.append(cromosoma(M,iteracion,poblacion[w1].w))
                poblacion_prima = np.append(poblacion_prima,poblacion[w1])
            else:
#                poblacion_prima.append(cromosoma(M,iteracion,poblacion[w2].w))
                poblacion_prima = np.append(poblacion_prima,poblacion[w2])
            
#        i += N
        
#        print("Tiempo parte 1:",var2-var)

        
# =============================================================================
#         PARTE 2: CRUCES. HACEMOS LA ESPERANZA MATEMATICA PARA DETERMINAR
#         EL NUMERO DE CRUCES QUE HAREMOS. SE CRUZARAN LOS num_cruces PRIMEROS
#         ELEMENTOS DE poblacion_prima .
#         
#         ESPERANZA MATEMATICA:
#         NUMERO ESPERADO DE CRUCES: Pc * N / 2
# =============================================================================
       
#        num_cruces = Pc * N / 2
#        num_cruces = int(num_cruces)

        j = 0
        poblacion_doble_prima = poblacion_prima

        for k in range(0,num_cruces): #Realizamos los cruces
            if boolean:
                hijo1,hijo2 = cruce_blx(poblacion_prima[j],poblacion_prima[j+1],0.3,iteracion)
            else:
                hijo1,hijo2 = cruce_aritmetico(poblacion_prima[j],poblacion_prima[j+1],iteracion)

#            print(hijo1,hijo2)
#            poblacion_doble_prima.append(cromosoma(M,iteracion,hijo1))
#            poblacion_doble_prima.append(cromosoma(M,iteracion,hijo2))
            poblacion_doble_prima[j] = cromosoma(M,iteracion,hijo1)
            poblacion_doble_prima[j+1] = cromosoma(M,iteracion,hijo2)
            j += 2
        
#        resto = range(0,num_cruces*2)
#        quitado = np.delete(poblacion_prima,resto,0)
#        poblacion_doble_prima = np.concatenate((poblacion_doble_prima,quitado),axis=0)

        
        i += num_cruces * 2
        
#        var2 = time()
#        print("Tiempo parte 2:",var2-var)

# =============================================================================
#         PARTE 3: MUTACIONES. HACEMOS LO MISMO QUE ANTES: CALCULAMOS LA ESPERANZA
#         DEL NUMERO DE MUTACIONES QUE HABRÁ. GENERAMOS UN NUMERO ALEATORIO Y SELECCIONAMOS
#         ALEATORIAMENTE LA FILA Y LA COLUMNA A MUTAR.
#         
#         ESPERANZA MATEMATICA:
#         NUMERO ESPERADO DE MUTACIONES: Pm * N * M
# =============================================================================
        
#        mutaciones = (int)(N * M * Pm)

        
        for j in range(0,mutaciones):
            np.random.seed(i)
            mu_next = r.randrange(N*M)
            pos_i = (int)(mu_next % N)
            pos_j = (int)(mu_next % M)
            
            poblacion_doble_prima[pos_i].w[pos_j] += np.random.normal(0.0, 0.3) #MUTACION
            
            if max(poblacion_doble_prima[pos_i].w) > 1 or min(poblacion_doble_prima[pos_i].w) < 0:
                poblacion_doble_prima[pos_i].w = corregir_pesos(poblacion_doble_prima[pos_i].w)
                
            poblacion_doble_prima[pos_i].comprobar(iteracion)
            
        i += mutaciones
        
#        print("Tiempo parte 3:",var2-var)
        
        
        
# =============================================================================
#         PARTE ADICIONAL: ELITISMO. CROMPROBAMOS SI SE HA PERDIDO EL MEJOR
#         CROMOSOMA INICIAL. SI SE HA PERDIDO, BUSCAMOS EL PEOR CROMOSOMA
#         ACTUAL Y LO INTERCAMBIAMOS POR EL MEJOR DE LA INICIAL.
# =============================================================================
        
        encontrado = False
        
        for crom in poblacion_doble_prima: #lo buscamos
            if crom == mejor_individuo :
                encontrado = True
                break
              
        if encontrado == False:
            calidad_peor = 100
            for k in range(0,len(poblacion_doble_prima)): #buscamos el peor
                if poblacion_doble_prima[k].calificacion < calidad_peor:
                    calidad_peor = poblacion_doble_prima[k].calificacion
                    peor_individuo = k
                    
            poblacion_doble_prima = np.delete(poblacion_doble_prima,peor_individuo)
            poblacion_doble_prima = np.append(poblacion_doble_prima,mejor_individuo)
            #intercambiamos
          
        poblacion = poblacion_doble_prima #reemplazo
#        var2 = time()
#        print("parte final:",var2-var)
#        despues = round(i/150,2)
#        caculo = (var2-var)*100/(despues-antes)
#        print("Estimado:",caculo/60,caculo/3600)
        
# =============================================================================
#     FINAL: BUSCAMOS EL MEJOR CROMOSOMA DE LA POBLACION FINAL
#     PARA DEVOLVERLO, Y EVALUARLO DESPUES CON EL TEST.
# =============================================================================
    mex = 0
    
    for k in range(1,N):
        if poblacion[k].calificacion > poblacion[mex].calificacion:
            mex = k
            
    return poblacion[mex].w




def genetico_estacionario(data,iteracion,boolean):
    datos,etiquetas = separar(data)
    poblacion = []
# =============================================================================
#     AG ESTACIONARIO:
#         BOOLEAN = CRUCE BLX ALFA
#         !BOOLEAN = CRUCE ARITMETICO PONDERADO
# =============================================================================
    N = 30 #TENEMOS 30 CROMOSOMAS
#    N = 2
    Est = 2
    M = len(datos[0])
#    calificaciones = []
    vacio = []
    for i in range(0,N): #GENERAMOS LA POBLACION INICIAL
        poblacion.append(cromosoma(M,iteracion,vacio))
#        calificaciones.append(poblacion[i].calificacion)
    poblacion = np.array(poblacion, cromosoma)

    Pm = 0.001
    Pc = 1
    Pcrom = Pm * M
    num_cruces = Pc * Est / 2
    num_cruces = int(num_cruces)
    i = N
    
    while i < 15000: #evaluaciones
#        antes = round(i/150,2)
#        print(antes,"%")
#        var = time()
# =============================================================================
#         PARTE 0: GUARDAMOS EL MEJOR CROMOSOMA POR SI LO PERDEMOS
# =============================================================================
#        mex = 0
#        
#        for k in range(1,N):
#            if poblacion[k].calificacion > poblacion[mex].calificacion:
#                mex = k
#                
#        mejor_individuo = poblacion[mex]
#        var2 = time()
#        print("timpo parte 0:",var2-var)
# =============================================================================
#         PARTE 1: TORNEO BINARIO. ELEGIMOS DOS CROMOSOMAS AL AZAR Y
#         SERAN LA NUEVA POBLACION
# =============================================================================
        
        poblacion_prima = []
#        aleatorios = r.sample(range(N),k=N)

        for j in range(0,Est): 
            w1 = r.randrange(N)
            w2 = r.randrange(N)
#            w1 = aleatorios[j]
#            w2 = aleatorios[N-j-1] 
            
            if poblacion[w1].calificacion > poblacion[w2].calificacion:
#                poblacion_prima.append(cromosoma(M,iteracion,poblacion[w1].w))
                poblacion_prima = np.append(poblacion_prima,poblacion[w1])
            else:
#                poblacion_prima.append(cromosoma(M,iteracion,poblacion[w2].w))
                poblacion_prima = np.append(poblacion_prima,poblacion[w2])
                
        
#        i += N
        
#        print("Tiempo parte 1:",var2-var)

        
# =============================================================================
#         PARTE 2: CRUCES. HACEMOS LA ESPERANZA MATEMATICA PARA DETERMINAR
#         EL NUMERO DE CRUCES QUE HAREMOS. SE CRUZARAN LOS num_cruces PRIMEROS
#         ELEMENTOS DE poblacion_prima .
#         
#         ESPERANZA MATEMATICA:
#         NUMERO ESPERADO DE CRUCES: Pc * N / 2
# =============================================================================
       
#        num_cruces = Pc * N / 2
#        num_cruces = int(num_cruces)

        j = 0
        poblacion_doble_prima = []

        for k in range(0,num_cruces): #Realizamos los cruces
            if boolean:
               hijo1,hijo2 = cruce_blx(poblacion_prima[j],poblacion_prima[j+1],0.3,iteracion)
            else:
                hijo1,hijo2 = cruce_aritmetico(poblacion_prima[j],poblacion_prima[j+1],iteracion)
            poblacion_doble_prima.append(cromosoma(M,iteracion,hijo1))
            poblacion_doble_prima.append(cromosoma(M,iteracion,hijo2))
            j += 2
        
            
#        resto = range(0,num_cruces*2)
#        quitado = np.delete(poblacion_prima,resto,0)
#        poblacion_doble_prima = np.concatenate((poblacion_doble_prima,quitado),axis=0)

        i += num_cruces * 2
#        var2 = time()
#        print("Tiempo parte 2:",var2-var)

# =============================================================================
#         PARTE 3: MUTACIONES. HACEMOS LO MISMO QUE ANTES: CALCULAMOS LA ESPERANZA
#         DEL NUMERO DE MUTACIONES QUE HABRÁ. GENERAMOS UN NUMERO ALEATORIO Y SELECCIONAMOS
#         ALEATORIAMENTE LA FILA Y LA COLUMNA A MUTAR.
#         
#         ESPERANZA MATEMATICA:
#         NUMERO ESPERADO DE MUTACIONES: Pm * N * M
# =============================================================================
        
#        mutaciones = (int)(N * M * Pm)
#        mutaciones = (int)mutaciones
        contador = 0
        for j in range(0,Est):
            aleatorio = np.random.rand(1,1)
            if aleatorio < Pcrom:
                contador += 1
                pos_j = r.randrange(M)
#                print(len(poblacion_doble_prima[j].w))
            
                poblacion_doble_prima[j].w[pos_j] += np.random.normal(0.0, 0.3) #MUTACION
            
                if max(poblacion_doble_prima[j].w) > 1 or min(poblacion_doble_prima[j].w) < 0:
                    poblacion_doble_prima[j].w = corregir_pesos(poblacion_doble_prima[j].w)
                    
                poblacion_doble_prima[j].comprobar(iteracion)
            
        i += contador
        
#        print("Tiempo parte 3:",var2-var)
        
        
        
# =============================================================================
#         PARTE ADICIONAL: ELITISMO. CROMPROBAMOS SI SE HA PERDIDO EL MEJOR
#         CROMOSOMA INICIAL. SI SE HA PERDIDO, BUSCAMOS EL PEOR CROMOSOMA
#         ACTUAL Y LO INTERCAMBIAMOS POR EL MEJOR DE LA INICIAL.
# =============================================================================
        
#        encontrado = False
#        
#        for crom in poblacion_doble_prima: #lo buscamos
#            if crom == mejor_individuo :
#                encontrado = True
#                break
#              
#        if encontrado == False:
#            calidad_peor = 100
#            for k in range(0,len(poblacion_doble_prima)): #buscamos el peor
#                if poblacion_doble_prima[k].calificacion < calidad_peor:
#                    calidad_peor = poblacion_doble_prima[k].calificacion
#                    peor_individuo = k
#                    
#            poblacion_doble_prima = np.delete(poblacion_doble_prima,peor_individuo)
#            poblacion_doble_prima = np.append(poblacion_doble_prima,cromosoma(M,iteracion,mejor_individuo.w))
            #intercambiamos
        
        peor_uno = 0
        peor_dos = 0
        
        for k in range(1,N):
            if poblacion[k].calificacion < poblacion[peor_uno].calificacion:
                peor_uno = k
                
        for k in range(1,N):
            if poblacion[k].calificacion < poblacion[peor_dos].calificacion and k != peor_uno:
                peor_dos = k
                       
        
        calificaciones = []
        calificaciones.append(poblacion[peor_uno].calificacion)
        calificaciones.append(poblacion[peor_dos].calificacion)
        
        calificaciones.append(poblacion_doble_prima[0].calificacion)
        calificaciones.append(poblacion_doble_prima[1].calificacion)
        
        diccionario = {}
        diccionario.update({calificaciones[0]:poblacion[peor_uno]})
        diccionario.update({calificaciones[1]:poblacion[peor_dos]})
        diccionario.update({calificaciones[2]:poblacion_doble_prima[0]})
        diccionario.update({calificaciones[3]:poblacion_doble_prima[1]})

        calificaciones.sort(reverse = True)
        borrado = False
#        poblacion = poblacion_doble_prima #reemplazo
        if diccionario.get(calificaciones[0]) != poblacion[peor_uno] and diccionario.get(calificaciones[0]) != poblacion[peor_dos]:
            poblacion = np.delete(poblacion,peor_uno)
            borrado = True
        if peor_dos > peor_uno:
            peor_dos = peor_dos -1

        if borrado and (diccionario.get(calificaciones[1]) != poblacion[peor_dos]):
            poblacion = np.delete(poblacion,peor_dos)
        elif not borrado and (diccionario.get(calificaciones[1]) != poblacion[peor_uno] and diccionario.get(calificaciones[1]) != poblacion[peor_dos]):
            poblacion = np.delete(poblacion,peor_dos)
            
        a = 0
        while len(poblacion) != N:
            poblacion = np.append(poblacion,diccionario.get(calificaciones[a]))
            a += 1
        
#        var2 = time()
#        print("parte final:",var2-var)
#        despues = round(i/150,2)
#        caculo = (var2-var)*100/(despues-antes)
#        print("Estimado:",caculo/60,caculo/3600)
        
# =============================================================================
#     FINAL: BUSCAMOS EL MEJOR CROMOSOMA DE LA POBLACION FINAL
#     PARA DEVOLVERLO, Y EVALUARLO DESPUES CON EL TEST.
# =============================================================================
    mex = 0
    
    for k in range(1,N):
        if poblacion[k].calificacion > poblacion[mex].calificacion:
            mex = k
            
    return poblacion[mex].w


def memetico_generacional(data,iteracion,boolean,pls,mejorado):
    datos,etiquetas = separar(data)
    poblacion = []
    contador = 0
# =============================================================================
#     AM GENERACIONAL:
#         BOOLEAN = CRUCE BLX ALFA
#         !BOOLEAN = CRUCE ARITMETICO PONDERADO
# =============================================================================
    N = 30 #TENEMOS 30 CROMOSOMAS
    M = len(datos[0])
#    calificaciones = []
    vacio = []
    for i in range(0,N): #GENERAMOS LA POBLACION INICIAL
        poblacion.append(cromosoma(M,iteracion,vacio))
#        calificaciones.append(poblacion[i].calificacion)
    poblacion = np.array(poblacion, cromosoma)
    
    Pm = 0.001
    Pc = 0.7
    num_cruces = Pc * N / 2
    num_cruces = int(num_cruces)
    i = N
    mutaciones = (int)(N * M * Pm)
    
    while i < 15000: #evaluaciones
        contador += 1
#        antes = round(i/150,2)
#        print(antes,"%")
#        var = time()
        
# =============================================================================
#         PARTE 0: GUARDAMOS EL MEJOR CROMOSOMA POR SI LO PERDEMOS
# =============================================================================
        mex = 0
        
        for k in range(1,N):
#            print(poblacion[k].calificacion)
            if poblacion[k].calificacion > poblacion[mex].calificacion:
                mex = k
                
        mejor_individuo = poblacion[mex]
#        print(mejor_individuo.calificacion)
#        var2 = time()
#        print("timpo parte 0:",var2-var)
# =============================================================================
#         PARTE 1: TORNEO BINARIO. ELEGIMOS DOS CROMOSOMAS AL AZAR Y
#         LO ANIADIMOS A LA NUEVA POBLACION.
# =============================================================================
        
        poblacion_prima = []
#        aleatorios = r.sample(range(N),k=N)
        
        for j in range(0,N): 
            w1 = r.randrange(N)
            w2 = r.randrange(N)
#            w1 = aleatorios[j]
#            w2 = aleatorios[N-j-1] 
            
            if poblacion[w1].calificacion > poblacion[w2].calificacion:
#                poblacion_prima.append(cromosoma(M,iteracion,poblacion[w1].w))
                poblacion_prima = np.append(poblacion_prima,poblacion[w1])
            else:
#                poblacion_prima.append(cromosoma(M,iteracion,poblacion[w2].w))
                poblacion_prima = np.append(poblacion_prima,poblacion[w2])
            
#        i += N
        
#        print("Tiempo parte 1:",var2-var)

        
# =============================================================================
#         PARTE 2: CRUCES. HACEMOS LA ESPERANZA MATEMATICA PARA DETERMINAR
#         EL NUMERO DE CRUCES QUE HAREMOS. SE CRUZARAN LOS num_cruces PRIMEROS
#         ELEMENTOS DE poblacion_prima .
#         
#         ESPERANZA MATEMATICA:
#         NUMERO ESPERADO DE CRUCES: Pc * N / 2
# =============================================================================
       
#        num_cruces = Pc * N / 2
#        num_cruces = int(num_cruces)

        j = 0
        poblacion_doble_prima = poblacion_prima

        for k in range(0,num_cruces): #Realizamos los cruces
            if boolean:
                hijo1,hijo2 = cruce_blx(poblacion_prima[j],poblacion_prima[j+1],0.3,iteracion)
            else:
                hijo1,hijo2 = cruce_aritmetico(poblacion_prima[j],poblacion_prima[j+1],iteracion)

#            print(hijo1,hijo2)
#            poblacion_doble_prima.append(cromosoma(M,iteracion,hijo1))
#            poblacion_doble_prima.append(cromosoma(M,iteracion,hijo2))
            poblacion_doble_prima[j] = cromosoma(M,iteracion,hijo1)
            poblacion_doble_prima[j+1] = cromosoma(M,iteracion,hijo2)
            j += 2
        
#        resto = range(0,num_cruces*2)
#        quitado = np.delete(poblacion_prima,resto,0)
#        poblacion_doble_prima = np.concatenate((poblacion_doble_prima,quitado),axis=0)

        
        i += num_cruces * 2
        
#        var2 = time()
#        print("Tiempo parte 2:",var2-var)

# =============================================================================
#         PARTE 3: MUTACIONES. HACEMOS LO MISMO QUE ANTES: CALCULAMOS LA ESPERANZA
#         DEL NUMERO DE MUTACIONES QUE HABRÁ. GENERAMOS UN NUMERO ALEATORIO Y SELECCIONAMOS
#         ALEATORIAMENTE LA FILA Y LA COLUMNA A MUTAR.
#         
#         ESPERANZA MATEMATICA:
#         NUMERO ESPERADO DE MUTACIONES: Pm * N * M
# =============================================================================
        
#        mutaciones = (int)(N * M * Pm)

        
        for j in range(0,mutaciones):
            np.random.seed(i)
            mu_next = r.randrange(N*M)
            pos_i = (int)(mu_next % N)
            pos_j = (int)(mu_next % M)
            
            poblacion_doble_prima[pos_i].w[pos_j] += np.random.normal(0.0, 0.3) #MUTACION
            
            if max(poblacion_doble_prima[pos_i].w) > 1 or min(poblacion_doble_prima[pos_i].w) < 0:
                poblacion_doble_prima[pos_i].w = corregir_pesos(poblacion_doble_prima[pos_i].w)
                
            poblacion_doble_prima[pos_i].comprobar(iteracion)
            
        i += mutaciones
        
#        print("Tiempo parte 3:",var2-var)
        
        
        
# =============================================================================
#         PARTE ADICIONAL: ELITISMO. CROMPROBAMOS SI SE HA PERDIDO EL MEJOR
#         CROMOSOMA INICIAL. SI SE HA PERDIDO, BUSCAMOS EL PEOR CROMOSOMA
#         ACTUAL Y LO INTERCAMBIAMOS POR EL MEJOR DE LA INICIAL.
# =============================================================================
        
        encontrado = False
        
        for crom in poblacion_doble_prima: #lo buscamos
            if crom == mejor_individuo :
                encontrado = True
                break
              
        if encontrado == False:
            calidad_peor = 100
            for k in range(0,len(poblacion_doble_prima)): #buscamos el peor
                if poblacion_doble_prima[k].calificacion < calidad_peor:
                    calidad_peor = poblacion_doble_prima[k].calificacion
                    peor_individuo = k
                    
            poblacion_doble_prima = np.delete(poblacion_doble_prima,peor_individuo)
            poblacion_doble_prima = np.append(poblacion_doble_prima,mejor_individuo)
            #intercambiamos
          
        poblacion = poblacion_doble_prima #reemplazo
#        var2 = time()
#        print("parte final:",var2-var)

        
        
        if contador == 10:
            contador = 0
            if not mejorado:
                for k in range(0,N):
                    aleatorio = np.random.rand(1,1)
                    if aleatorio < pls:
                        poblacion[k].w = busqueda_local(iteracion,poblacion[k].w)
                        poblacion[k].comprobar
                        i += 1
                   
            else:
                mejores = []
                while len(mejores) != 0.1 * N:
                    mex = 0
                    for k in range(1,N):
                        if poblacion[k].calificacion > poblacion[mex].calificacion and k not in mejores:
                            mex = k
                    mejores.append(mex)
                    
                for k in range(0,len(mejores)):
                    poblacion[mejores[k]].w = busqueda_local(iteracion,poblacion[mejores[k]].w)
                    poblacion[mejores[k]].comprobar
                    i += 1
         
#        despues = round(i/150,2)
#        caculo = (var2-var)*100/(despues-antes)
#        print(despues,"%. Estimado:",caculo/60,caculo/3600)
# =============================================================================
#     FINAL: BUSCAMOS EL MEJOR CROMOSOMA DE LA POBLACION FINAL
#     PARA DEVOLVERLO, Y EVALUARLO DESPUES CON EL TEST.
# =============================================================================
    mex = 0
    
    for k in range(1,N):
        if poblacion[k].calificacion > poblacion[mex].calificacion:
            mex = k
            
    return poblacion[mex].w
archivos = ['datos/colposcopy.arff','datos/ionosphere.arff','datos/texture.arff']
archivos = ['datos/texture.arff']


var = time()
for archivo in archivos:
    data, meta = loaddata(archivo)
#    print(data[0])
    datos,etiquetas = separar(data)
    M = len(datos[0])
    if archivo == 'datos/texture.arff':
        scaler = MinMaxScaler()
        scaler.fit(datos)
        datos = scaler.transform(datos)
    datos,etiquetas = shuffle(datos,etiquetas)
    data = shuffle(data)
    vara = time()
        
    for k in range(3,5):
        vara = time()
        print("AGG BLX EN",k,": ","MEDIA: ",comprobar(genetico_generacional(data,k,True),k))
        vara2 = time()
        print(vara2-vara,"segundos")
        print("AGG CA EN",k,": ","MEDIA: ",comprobar(genetico_generacional(data,k,False),k))
        vara = time()
        print(vara-vara2,"segundos")
        print("AGE BLX EN",k,": ","MEDIA: ",comprobar(genetico_estacionario(data,k,True),k))
        vara2 = time()
        print(vara2-vara,"segundos")
        print("AGE CA EN",k,": ","MEDIA: ",comprobar(genetico_estacionario(data,k,False),k))
        vara = time()
        print(vara-vara2,"segundos")
#        print("AMG 1 EN",k," MEDIA:",comprobar(memetico_generacional(data,k,True,1,False),k))
        vara2 = time()
#        print(vara2-vara,"segundos")
        print("AMG 0.1 EN",k," MEDIA:",comprobar(memetico_generacional(data,k,True,0.1,False),k))
        vara = time()
        print(vara-vara2,"segundos")
        print("AMG 0.1 mej EN",k," MEDIA:",comprobar(memetico_generacional(data,k,True,1,True),k))
        print(time()-vara,"segundos")
        
        
#print("E",np.concatenate((data[:,0:1],data[:,1:2]),axis=1))

#print("Calidad grupo:",0,comprobar(genetico_estacionario(data,0,False),0))
var2 = time()
print("Tiempo:",var2-var,"segundos.")
print("Tiempo:",(var2-var)/60,"minutos.")
