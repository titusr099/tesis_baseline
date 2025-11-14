from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

# Función que construye los pares (x, y) para un modelo seq2seq sobre grafos
# a partir de un DataFrame de series temporales multivariadas (tiempo × nodos).
# IMPORTANTE: Esta función NO construye el grafo/adjacency; solo arma las ventanas temporales.
def generate_graph_seq2seq_io_data(
        df,                        # pd.DataFrame con shape (num_samples, num_nodes). Index = timestamps.
        x_offsets,                 # np.ndarray de enteros NEGATIVOS y 0: desplazamientos de HISTORIA (p. ej., [-11,...,0]).
        y_offsets,                 # np.ndarray de enteros POSITIVOS: desplazamientos de FUTURO (p. ej., [1,...,12]).
        add_time_in_day=True,      # Si True, añade como feature la fase diaria ∈ [0,1) para cada timestamp.
        add_day_in_week=False,     # Si True, añade one-hot del día de la semana (7 dim) como feature exógena.
        scaler=None                # Parámetro no usado aquí; se deja por compatibilidad/extensión futura.
):
    
    """
    Generate samples from a multivariate time series dataframe using sliding windows.

    Parameters
    ----------
    df : pd.DataFrame
        Matriz tiempo × nodos (num_samples, num_nodes). El índice debe ser de tipo datetime64 con paso fijo.
    x_offsets : np.ndarray
        Desplazamientos (negativos y 0) para construir la secuencia de entrada X respecto al tiempo t.
        Ej.: [-11, -10, ..., 0] ⇒ 12 pasos de historia.
    y_offsets : np.ndarray
        Desplazamientos (positivos) para construir la secuencia objetivo Y respecto al tiempo t.
        Ej.: [1, 2, ..., 12] ⇒ 12 pasos de horizonte futuro.
    add_time_in_day : bool
        Si True, se concatena un canal adicional con la fase del día normalizada [0,1).
        Este canal es conocido para pasado y futuro (no hay data leakage).
    add_day_in_week : bool
        Si True, se concatena un canal one-hot de tamaño 7 con el día de la semana (0=lunes ... 6=domingo).
        También es una covariable conocida para pasado y futuro.
    scaler : Any
        (No utilizado aquí). En pipelines reales se usa para normalizar/estandarizar los valores del canal de velocidad.

    Returns
    -------
    x : np.ndarray
        Tensor de entradas con shape (N, Tx, num_nodes, F_in),
        donde N = número de ventanas generadas, Tx = len(x_offsets), F_in = nº de features de entrada.
    y : np.ndarray
        Tensor de objetivos con shape (N, Ty, num_nodes, F_out),
        donde Ty = len(y_offsets). En este script F_out coincide con F_in (p. ej., incluye time_of_day).
        *En entrenamiento normalmente se usa y[..., :1] si el objetivo es solo la velocidad.*
    """
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)


    # --- 1) Preparación de base de datos en 3D: (tiempo, nodos, features) -----------------------------
    # Convierte el DataFrame a un array 3D de shape (num_samples, num_nodes, num_features).
    num_samples, num_nodes = df.shape   #(34272, 207)
    data = np.expand_dims(df.values, axis=-1)   #(34272, 207, 1). El último eje es “características”: por ahora solo la velocidad (1 feature).
    # data_list acumulará todos los features (velocidad, time_of_day, day_of_week, etc.)
    data_list = [data]


    # --- 2) Feature opcional: "time of day" (fase del día) -------------------------------------------
    # Calcula para cada timestamp su fracción dentro del día en rango [0, 1),
    # p. ej., 00:00 -> 0.0, 12:00 -> 0.5, 23:30 -> ~0.979...
    if add_time_in_day:
        # time_of_day = (hora actual en segundos) / (24 horas en segundos)
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        # time_ind: array (34272,) con fracción del día [0,1)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
        # Esto mete una segunda feature por nodo: “en qué parte del día” está cada muestra.


    # --- 3) Feature opcional: one-hot del día de la semana -------------------------------------------
    # Si se activa, agrega un bloque de 7 canales con el indicador del día (uno en 1, resto en 0).
    if add_day_in_week:
        # Creamos un tensor de ceros: (num_samples, num_nodes, 7)
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week) # Se añaden 7 canales más (uno por día)


    #  --- 4) Concatenación final de features -----------------------------------------------------------
    # Unimos todos los canales en el eje de features: (tiempo, nodos, F_total).
    data = np.concatenate(data_list, axis=-1) # -> (34272, 207, 2) = (tiempo, nodos, features)
    # (feature 0 = velocidad; feature 1 = time_of_day)


     # --- 5) Construcción de las ventanas deslizantes (seq2seq) ---------------------------------------
    # x almacenará las secuencias de entrada; y almacenará las secuencias objetivo.
    x, y = [], []

    # min_t: primer índice 't' válido para poder mirar hacia atrás hasta el offset más negativo (p. ej., -11).
    # Si x_offsets = [-11,...,0], necesitamos t >= 11 para que t + (-11) no sea < 0.
    min_t = abs(min(x_offsets)) # e.g., 11

    # max_t (EXCLUSIVO): último índice 't' que permite mirar hacia adelante sin salirse del rango.
    # Si y_offsets = [1,...,12], el máximo 't' permitido es num_samples - 12 - 1, por lo que el 'stop'
    # del range es num_samples - 12 (exclusivo).
    max_t = abs(num_samples - abs(max(y_offsets)))  #e.g., 34272 - 12 = 34260 (límite superior exclusivo)
 
    # Recorremos t desde min_t (incluido) hasta max_t (excluido).
    # Para cada t:
    #   - X toma los tiempos t + x_offsets  ⇒ historia (p. ej., [-11..0] ⇒ 12 pasos).
    #   - Y toma los tiempos t + y_offsets  ⇒ futuro   (p. ej., [1..12] ⇒ 12 pasos).

    # Shapes por ejemplo:
    #   x_t: (Tx, num_nodes, F_total)  → (12, 207, 2)
    #   y_t: (Ty, num_nodes, F_total)  → (12, 207, 2)
    
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...] # Selección vectorizada de la historia alrededor de t
        y_t = data[t + y_offsets, ...] # Selección vectorizada del horizonte futuro alrededor de t
        x.append(x_t)
        y.append(y_t)

    # Apilamos todas las ventanas en un nuevo eje 0 para obtener tensores 4D.
    # N (número de ejemplos) = max_t - min_t. Con (34272 muestras, Tx=12, Ty=12) ⇒ N = 34260 - 11 = 34249.
    x = np.stack(x, axis=0) # -> # → (N, Tx, num_nodes, F_total)  p. ej., (34249, 12, 207, 2)
    y = np.stack(y, axis=0) # (num_samples, input_length, num_nodes, num_features)

    return x, y


# Orquesta la generación de tensores (x, y), realiza el split temporal (train/val/test)
# y guarda cada parte en archivos .npz comprimidos junto con los offsets usados.
def generate_train_val_test(args):
    # 1) Cargar el DataFrame de tráfico desde un HDF5 (clave por defecto: 'df').
    #    Espera una matriz (num_samples, num_nodes) con índice datetime a paso fijo (5 min en METR-LA).
    #df = pd.read_hdf(args.traffic_df_filename)   # Original: lee desde HDF5
    df = pd.read_csv(args.traffic_df_filename, parse_dates=[0], index_col=0) # Modificado: lee desde CSV

    # ----------------------------------------------------------------------------------------
    # 2) Definir los offsets temporales para entrada (x) y objetivo (y).
    #    Convención: 0 es la observación "anclada" en t; negativos miran hacia atrás; positivos, hacia adelante.
    #    x_offsets: [-11, -10, ..., 0]  → 12 pasos pasados (1 hora de historia con datos cada 5 minutos).
    #    y_offsets: [  1,   2, ..., 12] → 12 pasos futuros (1 hora de horizonte).
    # ----------------------------------------------------------------------------------------

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        # La línea comentada sugiere que podrían incluirse rezagos largos (día/semana) además de la ventana corta.
        np.concatenate((np.arange(-11, 1, 1),))    # [-11..0]
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))       # [1..12]

    # ----------------------------------------------------------------------------------------
    # 3) Construir los tensores X e Y a partir del DataFrame usando ventanas deslizantes (seq2seq).
    #    x: (N, Tx, num_nodes, input_dim)
    #    y: (N, Ty, num_nodes, output_dim)
    #    Donde típicamente input_dim/output_dim = 2 si se incluye 'time_of_day' (además de la velocidad).
    # ----------------------------------------------------------------------------------------

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,         # Añade canal 'fase del día' ∈ [0,1).
        add_day_in_week=False,        # No añade one-hot de día de la semana (7 canales).
    )

    # Log de verificación de shapes resultantes (útil para depuración).
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # ----------------------------------------------------------------------------------------
    # 4) Split temporal en train / val / test.
    #    Criterio: mantener el orden temporal (no mezclar), típico en series de tiempo.
    #    Proporciones aquí: 70% train, 10% val, 20% test (aprox. por redondeo).
    # ----------------------------------------------------------------------------------------
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]               # N = número total de ventanas generadas
    num_test = round(num_samples * 0.2)    # Último 20% para test (parte más reciente)
    num_train = round(num_samples * 0.7)   # Primer 70% para entrenamiento
    num_val = num_samples - num_test - num_train  # Resto (≈10%) para validación

    # train: primeros 'num_train' ejemplos (segmento más antiguo)
    x_train, y_train = x[:num_train], y[:num_train]
    
    # val: bloque intermedio inmediatamente posterior a train
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    
    # test: últimos 'num_test' ejemplos (segmento más reciente)
    x_test, y_test = x[-num_test:], y[-num_test:]

    # ----------------------------------------------------------------------------------------
    # 5) Guardado en disco: crea tres archivos .npz comprimidos: train.npz, val.npz, test.npz.
    #    Cada archivo incluye:
    #      - x: tensores de entrada
    #      - y: tensores objetivo
    #      - x_offsets, y_offsets: documentan los rezagos usados (guardados como columnas 2D)
    # ----------------------------------------------------------------------------------------
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),      # p. ej., data/train.npz
            x=_x,
            y=_y,
            # Guardar offsets con shape (len(offsets), 1) para mantener consistencia dimensional.
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    """
    Función principal del script.
    Su único propósito es llamar a la función que genera los datos de entrenamiento,
    validación y prueba (train/val/test) a partir del archivo bruto de tráfico.
    """
    # Llama a la función que realiza todo el procesamiento de datos:
    # 1. Leer el archivo metr-la.h5
    # 2. Construir ventanas seq2seq (x, y)
    # 3. Dividir en train, val y test
    # 4. Guardar los archivos .npz resultantes
    print("Generating training data")
    generate_train_val_test(args)


# Este bloque se ejecuta solo cuando el script es llamado desde línea de comandos,
# NO cuando se importa como un módulo en otro archivo.
if __name__ == "__main__":
    #  argparse permite capturar argumentos desde la terminal al ejecutar el script
    parser = argparse.ArgumentParser()

    # Argumento 1: directorio de salida donde se guardarán los archivos train.npz, val.npz, test.npz
    parser.add_argument(
        "--output_dir",               # nombre del parámetro esperado en la terminal
        type=str,                    # tipo de dato
        default="data/",            # valor por defecto (se usará si el usuario no lo especifica)
        help="Output directory."     # descripción que aparecerá si el usuario pide ayuda con -h
    )

    # Argumento 2: archivo de entrada con los datos brutos de tráfico en formato HDF5 (metr-la.h5)
    parser.add_argument(
        "--traffic_df_filename",     # nombre del parámetro
        type=str,
        default="data/metr-la.h5",   # ruta por defecto
        help="Raw traffic readings." # mensaje explicativo
    )

    # Parsea los argumentos proporcionados por la terminal y los guarda en `args`
    args = parser.parse_args()

    # Pasa esos argumentos a la función main
    main(args)
