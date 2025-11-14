from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    Construye la matriz de adyacencia (pesos del grafo) a partir de un DataFrame de distancias.
    El objetivo es transformar distancias físicas entre sensores en *similitudes* (pesos) usando
    un kernel Gaussiano, y luego umbralizar para obtener una matriz *dispersa* (sparse).

    Parámetros
    ----------
    distance_df : pd.DataFrame
        Debe tener exactamente tres columnas: [from, to, distance].
        - 'from' y 'to' son IDs de sensores (strings o ints coherentes con `sensor_ids`).
        - 'distance' es la distancia física (p. ej., metros/kilómetros) entre esos dos sensores.
          Solo se incluyen pares para los que hay medición; si falta, se asumirá "sin arista".
    sensor_ids : List[str] | List[int]
        Lista *ordenada* de IDs de sensores. Su orden define el orden de filas/columnas de la matriz.
    normalized_k : float (por defecto 0.1)
        Umbral de corte tras normalizar con el kernel Gaussiano: cualquier peso < normalized_k se hace 0.
        Esto reduce ruido y mejora eficiencia al hacer la matriz más rala.

    Returns
    -------
    sensor_ids : igual a la entrada
        Se devuelve por conveniencia (para guardar juntos).
    sensor_id_to_ind : Dict[id -> int]
        Mapa de ID de sensor a índice (posición en la matriz).
    adj_mx : np.ndarray de shape (num_sensors, num_sensors), dtype float32
        Matriz de adyacencia resultante (pesos ∈ [0,1]). Zeros en pares sin conexión o por debajo del umbral.
        *Nota*: Por defecto no hay autoconexiones (diagonal = 0) porque se inicializa con ∞ y exp(-∞)=0.
    """
    
    # ------------------------------------------------------------------------------------------
    # 1) Preparación: tamaño del grafo y matriz de distancias base
    # ------------------------------------------------------------------------------------------
    num_sensors = len(sensor_ids)

    # Matriz de distancias inicializada a ∞ (infinito) para denotar "sin conexión directa".
    # Posteriormente solo rellenamos las entradas (i, j) donde tenemos distancia conocida.
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf        # Con esto, exp(-∞) será 0, lo que equivale a "sin arista" tras el kernel.


    # ------------------------------------------------------------------------------------------
    # 2) Mapeo de IDs de sensores a índices (filas/columnas de la matriz)
    # ------------------------------------------------------------------------------------------
    # Este diccionario nos permite traducir 'from'/'to' (IDs) a índices numéricos en [0, num_sensors-1].
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    
    # ------------------------------------------------------------------------------------------
    # 3) Rellenar la matriz de distancias con los valores reales conocidos
    # ------------------------------------------------------------------------------------------
    # Recorremos cada fila del DataFrame de distancias:
    # - Si 'from' o 'to' no están en el vocabulario de sensor_ids, se ignora (salvaguarda).
    # - Si existen, colocamos la distancia en la celda correspondiente.
    #
    # *Direccionalidad*:
    #   Aquí solo se rellena dist[i, j]. Si tu CSV trae distancias asimétricas o solo una dirección,
    #   la matriz resultante será potencialmente asimétrica. Puedes simetrizar luego si lo necesitas.
    for row in distance_df.values:
        # row[0] -> from, row[1] -> to, row[2] -> distance
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]


    # ------------------------------------------------------------------------------------------
    # 4) Transformación de distancias a similitudes con un kernel Gaussiano
    # ------------------------------------------------------------------------------------------
    # El kernel Gaussiano:  A_ij = exp( - (d_ij / σ)^2 )
    #   - Distancias pequeñas -> A_ij cercano a 1 (conexión fuerte)
    #   - Distancias grandes  -> A_ij cercano a 0 (conexión débil)
    #   - Distancia ∞         -> exp(-∞) = 0 (sin conexión)
    #
    # Elegimos σ (sigma) como la desviación estándar de todas las distancias finitas.
    # Esto adapta la escala del kernel a la distribución real de distancias.
    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()      # solo valores finitos
    std = distances.std()


    # *Salvaguarda opcional*:
    # En escenarios extremos (todas distancias iguales), std podría ser 0. Aquí se asume que no ocurre.
    # Si te preocupa este caso, podrías hacer: std = max(std, 1e-6)

    # -----Para mi matriz si son todas las distancias iguales, entonces std=0 y da error en la siguiente línea por la división por 0.-----
    # so hacemos esto:
    if std == 0 or np.isclose(std, 0.0):
        # Binariza: arista presente -> 1.0; ausente (∞) -> 0.0
        adj_mx = np.where(np.isinf(dist_mx), 0.0, 1.0).astype(np.float32)
    else:
        # Aplica el kernel Gaussiano para convertir distancias en pesos de adyacencia.
        adj_mx = np.exp(-np.square(dist_mx / std))     # aplica el kernel elemento a elemento

    # ------------------------------------------------------------------------------------------
    # 5) (Opcional) Simetrización de la matriz de adyacencia
    # ------------------------------------------------------------------------------------------
    # Si quieres un grafo no dirigido (pesos simétricos), activa esta línea:
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    #
    # Dejarla comentada permite conservar direccionalidad si el CSV es direccional.

    # ------------------------------------------------------------------------------------------
    # 6) Umbralizado (sparsificación): eliminar conexiones débiles
    # ------------------------------------------------------------------------------------------
    # Cualquier peso menor que 'normalized_k' se fija a 0. Esto:
    #   - Reduce ruido de conexiones muy débiles (numéricamente pequeñas).
    #   - Hace la matriz más rala, mejorando rendimiento en modelos GNN.
    adj_mx[adj_mx < normalized_k] = 0

    # *Notas de diseño*
    #   Si tu modelo requiere self-loops, puedes forzarlas después: adj_mx[np.diag_indices(num_sensors)] = 1.0
    # - Unidades de distancia: asegúrate de que sean coherentes (todas en metros, todas en km, etc.).
    #   Cambiar las unidades escalara σ proporcionalmente y reescalará los pesos.
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='data/sensor_graph/graph_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data/sensor_graph/distances_la_2012.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/sensor_graph/adj_mat.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    normalized_k = args.normalized_k
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)



# La idea es convertir distancias físicas en pesos de conectividad, porque el tráfico se propaga entre sensores cercanos:
# Si hay tráfico detenido en el sensor A, probablemente afecte pronto al sensor B si están cerca.
# Si están lejos, la influencia es baja o nula.


# El modelo DCRNN realiza convoluciones sobre grafos, donde la información fluye desde nodos vecinos.
# El peso W_{ij} le dice al modelo cuánto debe tomar en cuenta el estado del sensor i para actualizar el estado del sensor j.