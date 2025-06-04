import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import label


#Normalize a image/matrix
def normalize_image(image):
    minim = np.amin(image)
    maxim = np.amax(image)
    
    if(minim != maxim):
       normalize_matrix = (image - minim) / (maxim - minim)
    elif(maxim != 0):
       normalize_matrix = image/maxim
    else:
       normalize_matrix = image
   
    return normalize_matrix

# Convert a matrix/image in a picture     
def matrix_to_picture(ventana, matrix, size):
    matrix = normalize_image(matrix)
    image = (255*matrix).astype(np.uint8)
    image = Image.fromarray(image).resize(size)
    picture = ImageTk.PhotoImage(image, master = ventana)
    
    return picture


# Coordinates changing functions
def matrix_to_image_coordinates(matrix_coords, matrix_width, matrix_height, image_width, image_height):
    """
    Converts matrix coordinates to image coordinates.

    Args:
        matrix_coords (tuple): Matrix coordinates in the format (row, column).
        matrix_width (int): Width of the matrix.
        matrix_height (int): Height of the matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Image coordinates in the format (x, y).
    """
    row, column = matrix_coords

    # Calculate the scaling factors to map matrix coordinates to image coordinates
    x_scale = image_width / matrix_width
    y_scale = image_height / matrix_height

    # Calculate the corresponding image coordinates
    x = column * x_scale
    y = row * y_scale

    # Make sure the coordinates are within the image bounds
    x = max(0, min(x, image_height - 1))
    y = max(0, min(y, image_width - 1))

    return x, y

def image_to_matrix_coordinates(image_coords, image_width, image_height, matrix_width, matrix_height):
    """
    Converts image coordinates to matrix coordinates.

    Args:
        image_coords (tuple): Image coordinates in the format (x, y).
        matrix_width (int): Width of the matrix.
        matrix_height (int): Height of the matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Matrix coordinates in the format (row, column).
    """
    x, y = image_coords

    # Calculate the scaling factors to map image coordinates to matrix coordinates
    x_scale = matrix_width / image_width
    y_scale = matrix_height / image_height

    # Calculate the corresponding matrix coordinates
    column = int(x * x_scale + 0.5) # round to integer
    row = int(y * y_scale + 0.5) # round to integer

    # Make sure the coordinates are within the matrix bounds
    column = max(0, min(column, matrix_height - 1))
    row = max(0, min(row, matrix_width - 1))

    return row, column


#Save a matrix as .png (image)
def save_matrix_as_image(matrix, name, quality = 16, _format = '.png'):
    
    matrix = normalize_image(matrix)
    matrix = matrix*(2**quality-1)
    
    if quality == 16:
        matrix = np.uint16(matrix)
    if quality == 8:
        matrix = np.uint8(matrix)
        
    image = Image.fromarray(matrix)
    image.save(name + _format)

# <-------------------------------------------------------------------------Image processing------------------------------------------------------------------------->

def isolate_particles(pred_mask, threshold=0.5):
    binary_mask = (pred_mask > threshold).astype(np.uint8)  # Convierte la máscara de probabilidades en una máscara binaria
    labeled_mask, num_particles = label(binary_mask) # Function from scipy.ndimage, busca grupos de píxeles conectados entre sí
    return labeled_mask, num_particles
    
def get_indices_box(indices):
    """Calcula los límites (r_min, r_max, c_min, c_max) a partir de los índices de probabilidad."""
    rows, cols = indices
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (np.min(rows), np.max(rows), np.min(cols), np.max(cols))

def box_contains_any_center(index_box, zone_centers):
    """
    Dada una caja de índices (r_min, r_max, c_min, cmax) y una lista de centros (x, y),
    retorna True si al menos uno de esos centros está contenido en la caja.
    Se asume que x corresponde a la fila y y a la columna.
    """
    rmin, rmax, cmin, cmax = index_box
    for center in zone_centers:
        x, y = center  # Solo se usan las dos primeras coordenadas
        if rmin <= x <= rmax and cmin <= y <= cmax:
            return True
    return False

def rectangle_distance_indices(box1, box2):
    """
    Calcula la distancia entre dos cajas definidas por índices.
    Cada caja es (r_min, r_max, c_min, c_max). La distancia se define a partir del gap horizontal y vertical.
    Si se solapan en alguna dirección, el gap es 0 en esa dirección.
    """
    rmin1, rmax1, cmin1, cmax1 = box1
    rmin2, rmax2, cmin2, cmax2 = box2
    gap_h = max(0, max(cmin1, cmin2) - min(cmax1, cmax2))
    gap_v = max(0, max(rmin1, rmin2) - min(rmax1, rmax2))
    return np.sqrt(gap_h**2 + gap_v**2)

def merge_index_boxes(box1, box2):
    """Fusiona dos cajas (r_min, r_max, c_min, c_max) tomando la unión de sus límites."""
    rmin1, rmax1, cmin1, cmax1 = box1
    rmin2, rmax2, cmin2, cmax2 = box2
    return (min(rmin1, rmin2), max(rmax1, rmax2),
            min(cmin1, cmin2), max(cmax1, cmax2))

def merge_nearby_index_boxes(boxes, merge_threshold):
    """
    Fusiona cajas de índices que estén separadas por una distancia (entre sus límites)
    menor que merge_threshold. Se iterará hasta que ya no se puedan fusionar.
    """
    merged_boxes = boxes.copy()
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(merged_boxes)
        for i in range(len(merged_boxes)):
            if used[i]:
                continue
            current = merged_boxes[i]
            for j in range(i+1, len(merged_boxes)):
                if used[j]:
                    continue
                if rectangle_distance_indices(current, merged_boxes[j]) < merge_threshold:
                    current = merge_index_boxes(current, merged_boxes[j])
                    used[j] = True
                    merged = True
            new_boxes.append(current)
            used[i] = True
        merged_boxes = new_boxes
    return merged_boxes

def index_box_to_square(index_box, image_shape):
    """
    Convierte una caja de índices (r_min, r_max, c_min, c_max) en un cuadrado.
    Se calcula el lado como el máximo entre la altura y el ancho y se centra.
    """
    rmin, rmax, cmin, cmax = index_box
    height = rmax - rmin
    width = cmax - cmin
    side = max(height, width)
    center_r = (rmin + rmax) // 2
    center_c = (cmin + cmax) // 2
    half_side = side // 2
    start_row = max(center_r - half_side, 0)
    start_col = max(center_c - half_side, 0)
    if start_row + side > image_shape[0]:
        start_row = image_shape[0] - side
    if start_col + side > image_shape[1]:
        start_col = image_shape[1] - side
    return (start_row, start_col, side)

def create_boxes(image, min_particle_size, th=0.2, num_particles=None, labeled_mask=None, zones=None, pred_mask=None):
    # Extraer cajas de índices de cada partícula de la máscara
    index_boxes = []
    min_particle_size = 0  # Tamaño mínimo en píxeles para aceptar una partícula
    for x in range(1, num_particles + 1):
        particle_indices = np.where(labeled_mask == x)
        ibox = get_indices_box(particle_indices)
        if ibox is not None:
            # Descartar si la altura o anchura es menor que el tamaño mínimo
            if (ibox[1] - ibox[0] >= min_particle_size) and (ibox[3] - ibox[2] >= min_particle_size):
                index_boxes.append(ibox)
    
    # Construir una lista de centros a partir de 'zones' (tomando solo x,y)
    zone_centers = []
    for key, centers in zones.items():
        for center in centers:
            x, y, _ = center
            if pred_mask[int(x), int(y)] >= th:  # Sólo si cumplen el threshold
                zone_centers.append((x, y))
    
    # Filtrar las cajas de índices: conservar sólo aquellas que contienen al menos un centro
    filtered_index_boxes = [ibox for ibox in index_boxes if box_contains_any_center(ibox, zone_centers)]
    
    # Fusionar las cajas de índices basándose en la distancia entre sus límites
    merge_distance_threshold = 20  # Umbral en píxeles para fusionar
    merged_index_boxes = merge_nearby_index_boxes(filtered_index_boxes, merge_distance_threshold)
    
    # Convertir cada caja de índices fusionada en un cuadrado final
    final_bboxes = [index_box_to_square(ibox, image.shape) for ibox in merged_index_boxes]
    
    # (Internamente se calculan centros, pero en la imagen final solo dibujamos cuadrados uniformes)
    centers_list = [(r + side/2, c + side/2) for (r, c, side) in final_bboxes]
    print("Número de nanopartículas detectadas después de la fusión:", len(final_bboxes))
    print("Centros (fila, columna):", centers_list)
    
    return centers_list, final_bboxes
