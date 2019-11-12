from scipy.io import arff
import numpy as np 
import random as r
from time import time
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

np.random.seed(0)

def loaddata(path):
    f = path
    data, meta = arff.loadarff(f)
    df_data = pd.DataFrame(data)
    data = df_data.values
    

#        print("Escalado de valores.")
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

def separar(data):
    datos = []  
    etiquetas = []
    
    for data1 in data:
#        print(data1[-1])
        etiquetas.append(data1[-1])
        datos.append(data1[0:-1])
        
    etiquetas = np.array(etiquetas, np.float64)  #Conversion al tipo correcto
    datos = np.array(datos, np.float64)
    
    return datos,etiquetas
#funcion de distancia
def multiplica_pesos(x,pesos):
    
    M = len(x[0]) 
    data = x[:,0:M]
    for i in range(0,len(x)):
        for j in range(0,len(x[0])-1):
            if pesos[j] >= 0.2:
                data[i][j] = x[i][j]*pesos[j]
            else:
                data[i][j] = 0

    return data
            

#funcion para cargar los datos
def carga_datos(data,iteracion):
#    M = len(data[0]) -1
    N = len(data)
    tam = N // 5
    i = iteracion
    
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

#funcion que cuenta cuantos pesos tienen valor inferior a 0.2
def califica_pesos(pesos):
    contador = 0
    for p in pesos:
        if p <= 0.2:
            contador += 1
    return contador

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

#funcion para el vecino mas cercano
def amigo_cercano(data,x,pesos,boolean):
    caracter = x[-1]
    cadena = []
    
    tree = KDTree(data, leaf_size=1)
    
    x = x.reshape(1,-1)
    i = 2
    
    while True:
        dis,ind = tree.query(x, k=i)
#        ind2 = ind
        ind = ind[0][-1]
#        print(dis,(x==data[ind]).all(),ind2)
#        break
        if (boolean and data[ind][-1] == caracter) or (not boolean and data[ind][-1] != caracter):
            cadena = data[ind]
            break
        else:
            i += 1

    
    return cadena

#calcula los pesos en funcion del enemigo y amigo
def calcula_nuevos_pesos(pesos,x,amigo,enemigo):
    w = pesos
    #seria pesos + (enemigo - ejemplo) - (amigo - ejemplo)
    # lo que es enemigo - amigo
    for i in range(0,len(amigo)-1):
        w[i] = pesos[i] + abs(enemigo[i]) - abs(amigo[i])

    return w
    
#truncamos los pesos  
def corregir_pesos(pesos):
    w = []
    maximo = max(pesos)
    for p in pesos:
        if p < 0:
            w.append(0)
        elif p > 1:
            w.append(p/maximo)
        else:
            w.append(p)

    return w

#algorimtmo GREEEDY
def greedy(i):

    pesos = np.zeros(M,np.float64)
    x,y = carga_datos(data,i)
    for x1 in x:
        target_amigo = amigo_cercano(x,x1,pesos,True)
        target_enemigo = amigo_cercano(x,x1,pesos,False)
        pesos = calcula_nuevos_pesos(pesos,x1,target_amigo,target_enemigo)
        pesos = corregir_pesos(pesos)
        
#    pesos = corregir_pesos(pesos)
    return pesos

#pesos aleatorios
def inicia_pesos():
    np.random.seed(1)
    
    w = []
    
    for i in range(0,M):
        numero = r.randrange(100)
        w.append(numero/100)
        
    return w
    
def comprobar_bl(pesos,iteracion):
#def comprobar_(pesos,iteracion):
    _pesos = np.copy(pesos)
    _pesos[_pesos < 0.2] = 0
    
    train,test = carga_datos(data,iteracion) #no hacemos nada con la y.
    x_train,y_train = separar(train)
            
    x_train = (x_train*_pesos)[:,_pesos > 0.2]

    tree = KDTree(x_train)

    dis,vecinos = tree.query(x_train,k=2)
    vecinos = vecinos[:,1]
    aciertos = np.mean( y_train[vecinos] == y_train)*100
    
    calif = califica_pesos(pesos) * 100 / len(pesos)
  

    return (aciertos + calif) /2

#BUSQUEDA LOCAL
def busqueda_local(j):
    #solucion inicial:
    pesos = inicia_pesos()
#    pesos = greedy()
#    pesos = np.zeros(M,np.float64)

    desviacion = 0.3
    O = len(pesos)
    
#    train,test = carga_datos(data,j)
#    datos_train,etiquetas_train = separar(train)
    calidad = comprobar_bl(pesos,j)
    iters = 1
    no_mejora = 0
    while iters < 15000 and no_mejora < 20*O :

        for i in range(0,O):
            prev = pesos[i]
            valor = np.random.normal(0,desviacion)
            pesos[i] = np.clip(pesos[i] + valor,0,1)
            
            calidad1 = comprobar_bl(pesos,j)
#            print(calidad1)
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

def k_NN(data_training, tags_training, w, data_test = None, tags_test = None, is_training = True):
	w_prim = np.copy( w )
	w_prim[w_prim < 0.2] = 0.0
	eliminated = w_prim[w_prim < 0.2].shape[0]
	hit = 0
	hit_rate = 0.0

	data_training_mod = (data_training*w_prim)[:, w_prim > 0.2]

	tree = KDTree(data_training_mod)
	if is_training:
		nearest_ind = tree.query(data_training_mod, k=2, return_distance=False)[:,1]
		hit_rate = np.mean( tags_training[nearest_ind] == tags_training )
	else:
		data_test_mod = (data_test*w_prim)[:, w_prim > 0.2]
		nearest_ind = tree.query(data_test_mod, k=1, return_distance=False)
		for i in range(nearest_ind.shape[0]):
			if tags_training[nearest_ind[i]] == tags_test[i]:
				hit += 1

		hit_rate = hit/data_test_mod.shape[0]


	reduction_rate = eliminated/len(w)

	f = (hit_rate + reduction_rate)* 0.5

	return f, hit_rate, reduction_rate

def local_search(data, tags,iteracion):
	w = np.random.uniform(0.0,1.0,data.shape[1])
	max_eval = 15000
	max_neighbors = 20*data.shape[1]
	n_eval = 0
	n_neighbors = 0
	variance = 0.3
	mean = 0.0
	class_prev = comprobar_bl(w,iteracion)


	while n_eval < max_eval and n_neighbors < max_neighbors:
		for i in range(w.shape[0]):
			n_eval += 1
			prev = w[i]
			w[i] = np.clip(w[i] + np.random.normal(mean, variance), 0, 1)
			class_mod = comprobar_bl(w,iteracion)

            
			if(class_mod > class_prev):
				n_neighbors = 0
				class_prev = class_mod
				break
			else:
				w[i] = prev
				n_neighbors += 1

	"""
	for i in range(len(w)):
		plt.bar(i,w[i])
	plt.show()
	"""

	return w
archivos = ['datos/colposcopy.arff','datos/ionosphere.arff','datos/texture.arff']
#archivos = ['datos/texture.arff']

var = time()

for archivo in archivos:
    data, meta = loaddata(archivo)
#    print(data[0])
    datos,etiquetas = separar(data)
#    if archivo == 'datos/texture.arff':
    scaler = MinMaxScaler()
    scaler.fit(datos)
    datos = scaler.transform(datos)

    datos,etiquetas = shuffle(datos,etiquetas)
    data = shuffle(data)
#    data = shuffle(data)
#    print(data[0])
# =============================================================================
#     if archivo == 'datos/texture.arff':
#         scaler = MinMaxScaler()
#         scaler.fit(data)
#         data = scaler.transform(data)
# =============================================================================
#    print(etiquetas)
    M = len(data[0]) -1
    N = len(data)
    tam = N // 5

    pesos = np.ones(M,np.float64)
    
    for i in range(0,5):
        
        var1 = time()
        print("KNN, particion ",i,": ",comprobar(pesos,i,True),"%")
        
        var2 = time()
        print("Tiempo: ",var2-var1)
    
    for i in range(0,5):
        var2 = time()
#        print("GREEDY, particion ",i,": ",comprobar(greedy(i),i,True),"%")
        
        var3 = time()
#        print("Tiempo: ",var3-var2)
    
    for i in range(0,5):
        var3 = time()
#        training,test = carga_datos(data,i)
#        datos_tr,etiquetas_tr = separar(training)
#        print(etiquetas_tr)
#        print("BUSQUEDA LOCAL, particion ",i,": ",comprobar(busqueda_local(i),i),"%")
        
        var4 = time()
#        print("Tiempo: ",var4-var3)

print("Tiempo TOTAL: ",time()-var)
