Introduccion
================

Que es el ``Wrangler`` ? 
-------------------------
Es una libreria interna que hace mas facil llevar a produccion los modelos 
de machine learning desarrollados para los clientes. Nos permite trabajar en 
notebooks y que no sea un incordio pasarlo a archivos .py.

Como funciona el ``Wrangler``?
--------------------------------
*Long story short* : Hay una clase central en la libreria que es el ``Wrangler`` este 
va a guardar los dataset y los nodos, asi como tambien los metodos necesarios 
para ejecutar el pipeline. Si, porque al final del dia, el wrangler solo es una 
forma de guardar  una serie pasos ordenados que nos permita replicar un proceso 
de principio a fin.

Una vez que cargamos todos los dataset y todos los pasos del pipeline (ACA nodos), 
aplicamos el metodo run() para ejectuar todo. Facil no? JA 


Que necesita el ``Wrangler``?
---------------------------------


Para funcionar el ``Wrangler`` requiere 2 inputs valiosos: datasets y nodos. 
Los datasets se guardan en el ``DataCatalog`` y los nodos en el ``Pipeline``, 
estos dos atributos son importantes para saber donde estamos parados. 

Para cargar un dataset el ``Wrangler`` nos provee con un metodo llamado `add_dataset()` 
el cual recibe como argumento un "tipo de dato dataset" y un nombre para 
identificar al mismo (Mas adelante te explico un poco sobre eso). 

Por otro lado, si queremos cargar un nodo debemos usar el metodo `add_node()`, 
en el guardamos el nombre del paso, la transformacion que realiza, el data 
set o los dataset que usa y el/los ouputs (Mas adelante tambien hablamos sobre esto). 

Una vez que cargamos todos los datasets y los nodos lo que necesitamos es ejecutar el metoro `run()`
o directamente ejecutar el wrangler como una funcion. 
Que hace esto ? Ejecuta los pasos que nosotros hemos guardado en el Pipeline leyendo 
y guardando desde el DataCatalog todos los datasets declarados.


.. code-block:: python

    from wrangler.core import Wrangler
    wrangler = Wrangler()



Que es un ``Node``? 
-------------------------
Es nuestra manera de almacenar pasos en el ``Wrangler``. 
En definitiva nuestra intencion final es tomar uno o varios datasets y procesarlo 
en multiples pasos que sean reproducibles a futuro. Los pasos se van guardando en 
nodos y estos en un ``Pipeline``.

Que partes tiene un ``Node``? Bueno los nodos tienen: 

1. `name`:, es el nombre del paso que estamos ejecutando. Debe ser unico para cada
   nodo ya que si duplicamos nombres, los nodos se reemplazaran.
   
2. `inputs`: puede ser uno o varios dataset. Por ejemplo puede ser data set que contenga 
   los features para aplicar un standard scaler o el de features y target para fitear un modelo. 
   Como se le indica cual es el dataset que hay que usar? Se indica con el nombre con el que 
   esta identificaremos a ese dataset en el el ``DataCatalog``.

3. `outputs`: uno o varios ouputs de la misma manera el ouput puede ser X_test y X_train si en el nodo 
   ejecutamos un train test split o simplimente un data frame con predicciones si estamos 
   haciendo un predict. Es muy importante poner un nombre claro al ouput ya que asi es como 
   nos vamos a referir a el en el ``DataCatalog``.

4. `func`: Una funcion de python. Es la accion que se realiza en ese modo, puede ser desde cambiar el 
   nombre a una columna, seleccionar el dtype , o incluso crear el modelo. 


.. code-block:: python

    from wrangler.core import Wrangler
    wrangler = Wrangler()

    def my_func(a, b):
        return a + b 

    wrangler.add_node(
        func=my_func,
        inputs=['a','b'],
        outputs=['c'],
    )



Que es el ``DataCatalog``? 
-----------------------------

El ``DataCatalog`` es donde guardamos todos los dataset, es un atributo de la instancia 
del wrangler y se puede acceder como : `wrangler.catalog`. Aca no solo 
se guardan los datasets que cargamos (ver celda siguiente), sino tambien aquellos 
dataset intermedios que se van a generar a traves de los nodos.

Es muy poco usual que utilicemos el ``DataCatalog`` directamente, sino que 
es un objeto que utiliza internamente el ``Wrangler`` para almacenar los inputs
y outputs de los nodos.


Que podemos cargar en el ``DataCatalog``?
---------------------------------------------

Podemos importar un monton de tipos de datos, para ver mas sobre eso leer la 
documentacion(en especial el modulo de `data.datasets`). Entre los archivos mas comunes 
tenemos `CSV`, `Excel`, `Parquet`, `Json`, `Yaml`, `Pickle`. 

.. note::

   Tener cuidado cuando trabajamos con datasets leidos desde disco/s3 ya que si los
   modificamos y los volvemos a guardar, se modificarar el archivo original y no 
   sera posible la ejecutar nuevamente la misma modificacion.


Cada tipo de dataset tendra sus propios parametros para lectura y escritura. El mas
sencillo ``MemoryDataset``, directamente almacena en la memoria ram la data. En otros
casos, con indicar la ruta al archivo, alcanza, como por ejemplo, ``JSONDataset``, 
``PickleDataset``. Para los archivos .xlsx, .csv y .parquet, utilizamos la API de Pandas,
por lo que todos los parametros posibles de esta api se podrian especificar.

.. note::

   Un dato, es importante ir "pisando" los dataset entre nodos intermedios, ya que 
   sino nos va a explotar la RAM. Si tenemos muchos pasos intermedios, por ejemplo 
   cambiar de nombre una columna y luego cambiar de dtype otra, hacer que los nomnbres 
   del outputs del primer paso coincidan con los outputs del segundo (asi no duplicamos 
   los dataset en memoria).


Para un ejemplo de como agregar un dataset al data catalog ver celda 
de abajo.

.. code-block:: python

    from wrangler.data.datasets import MemoryDataset

    df_catalog = MemoryDataset(data =  df_train)
    wrangler.add_dataset(name = 'primer_input' , dataset =  df_catalog)



Ejemplito End-to-End
-----------------------

.. code-block:: python

    # import al necessary libraries
    from wrangler.core import Wrangler
    from wrangler.data.datasets import CSVDataset
    from wrangler.data.datasets import PickleDataset
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    # create wrangler
    wrangler = Wrangler()

    # features of the problem are saved in a file called features.csv
    features = CSVDataset(filename = 'data/features.csv')
    wrangler.add_dataset(name = 'features', dataset = features)

    # target of the problem is saved in a file called target.csv
    target = CSVDataset(filename = 'data/target.csv') 
    wrangler.add_dataset(name = 'target', dataset = target)

    # the model, once is trained will be saved as a pickle file.
    linear_model_ds = PickleDataset(filename = 'models/linear_regression.pkl')   
    wrangler.add_dataset(name = 'linear_model', dataset = linear_model_ds)


    def split_train_test(X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    wrangler.add_node(
        func=split_train_test,
        inputs=['features','target'],
        outputs=["X_train", "X_test", "y_train", "y_test"],
        func_kwargs = {"test_size":0.2}
    )

    def train_linear_model(X, y):
        lm = LinearRegression()
        return lm.fit(X,y)

    wrangler.add_node(
        func=train_linear_model,
        inputs=['X_train','y_train'],
        outputs=['linear_model'],
    )

    def predict_linear_model(lm, X):
        return lm.predict(X)

    wrangler.add_node(
        func=predict_linear_model,
        inputs=['linear_model','X_test'],
        outputs=['y_pred'],
    )

    def eval_linear_model(y_true, y_pred):
        return r2_score(y_true, y_pred)

    wrangler.add_node(
        func=eval_linear_model,
        inputs=['y_test','y_pred'],
        outputs=['r2'],
    )

    # Now we can run the wrangler.run() or alternatively wrangler()
    wrangler.run()


