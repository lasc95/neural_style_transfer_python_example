# Importa las bibliotecas necesarias
import numpy as np
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

# Define las rutas a las imágenes de contenido y estilo
content_image_path = 'ruta/imagen/de/contenido.jpg'
style_image_path = 'ruta/imagen/estilo.jpg'

# Carga la imagen de contenido y obtiene sus dimensiones
width, height = load_img(content_image_path).size
# Define la altura de la imagen generada
img_height = 400
# Calcula la anchura de la imagen generada manteniendo la relación de aspecto
img_width = int(width * img_height / height)

# Define una función para preprocesar las imágenes
def preprocess_image(image_path):
    # Carga la imagen y la redimensiona al tamaño objetivo
    img = load_img(image_path, target_size=(img_height, img_width))
    # Convierte la imagen en un array de numpy
    img = img_to_array(img)
    # Añade una dimensión extra al principio del array
    img = np.expand_dims(img, axis=0)
    # Realiza el preprocesamiento específico para el modelo VGG19
    img = vgg19.preprocess_input(img)
    # Devuelve la imagen preprocesada
    return img

# Define una función para deshacer el preprocesamiento de las imágenes
def deprocess_img(x):
    # Redimensiona la imagen al tamaño original
    x = x.reshape((img_height, img_width, 3))
    # Deshace la substracción de la media que se hizo durante el preprocesamiento
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # Asegura que todos los valores de los píxeles estén en el rango válido para una imagen de 8 bits (0-255)
    x = np.clip(x, 0, 255).astype('uint8')
    # Devuelve la imagen deprocesada
    return x

# Carga y preprocesa las imágenes de contenido y estilo
content_image = K.variable(preprocess_image(content_image_path))
style_image = K.variable(preprocess_image(style_image_path))
# Crea un placeholder para la imagen de combinación (la imagen generada)
combination_image = K.placeholder((1, img_height, img_width, 3))

# Combina las tres imágenes en un solo tensor
input_tensor = K.concatenate([content_image, style_image, combination_image], axis=0)

# Crea un modelo VGG19 preentrenado con los pesos de ImageNet
model = vgg19.VGG19(input_tensor, weights='imagenet', include_top=False)

# Define varias funciones de pérdida
def content_loss(base, combination):
    # Calcula la pérdida de contenido como la suma de los cuadrados de las diferencias entre las características de la imagen base y las de la imagen de combinación
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    # Calcula la matriz de Gram de una imagen
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    # Calcula la pérdida de estilo como la suma de los cuadrados de las diferencias entre las matrices de Gram de la imagen de estilo y la imagen de combinación
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(X):
    # Calcula la pérdida de variación total, que se utiliza para suavizar la imagen
    a = K.square(X[:, img_height - 1, :img_width - 1, :] - X[:, 1:, :img_width - 1, :])
    b = K.square(X[:, img_height - 1, :img_width - 1, :] - X[:, 1:, :img_height - 1, :])
    return K.sum(K.pow(a+b, 1.25))

# Crea un diccionario que mapea los nombres de las capas a las salidas de las capas
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# Define la capa de contenido y las capas de estilo
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_con1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
# Define los pesos para las diferentes componentes de la pérdida
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# Inicializa la pérdida total a 0
loss = K.variable(0.)
# Obtiene las características de la capa de contenido
layer_features = outputs_dict[content_layer]
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
# Añade la pérdida de contenido a la pérdida total
loss += content_weight * content_loss(content_image_features, combination_features)

# Añade la pérdida de estilo para cada capa de estilo a la pérdida total
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

# Añade la pérdida de variación total a la pérdida total
loss += total_variation_weight * total_variation_loss(combination_image)

# Define los gradientes de la pérdida con respecto a la imagen de combinación
grads = K.gradients(loss, combination_image)[0]
# Crea una función de Keras que toma una imagen de entrada y devuelve la pérdida y los gradientes
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# Define una clase Evaluador que calcula la pérdida y los gradientes en un solo paso mientras se ejecuta el algoritmo de optimización
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
# Crea una instancia de la clase Evaluador
evaluator = Evaluator()

# Preprocesa la imagen de contenido y la aplana
x = preprocess_image(content_image_path)
x = x.flatten()

# Ejecuta el algoritmo de optimización
for i in range(10):
    print('Start of iteration ', i)
    start_time = time.time()
    # Usa el algoritmo L-BFGS para minimizar la pérdida
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print('current loss value ', min_val)
    # Deprocesa la imagen generada y la guarda
    img = deprocess_img(x.copy())
    fname = 'output_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as ', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
