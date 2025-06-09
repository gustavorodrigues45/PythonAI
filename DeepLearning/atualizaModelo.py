######################################################
###            D E E P   L E A R N I N G           ###
######################################################
### Exemplo de Deep Learning utilizando MNIST      ###
### Prof. Filipo Mor - 02 de junho de 2025         ###
######################################################
### retreina o modelo, porém, com imagens geradas
### pelo usuário

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Defina o caminho para o diretório principal das suas imagens
data_dir = 'meus_digitos'  # Substitua pelo caminho real

# Parâmetros
image_size = (28, 28)
batch_size = 32  # Ajuste conforme necessário

# Crie um ImageDataGenerator para carregar e pré-processar as imagens
# ImageDataGenerator lida com o redimensionamento, normalização e divisão
datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza os valores dos pixels para entre 0 e 1
    validation_split=0.2  # Usar 20% dos dados para validação
)

# Carregue os dados de treinamento
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',  # Garante que as imagens sejam carregadas em escala de cinza
    class_mode='sparse_categorical',  # Labels são inteiros (0-9)
    subset='training',  # Seleciona a parte de treinamento
    seed=123  # Para reprodutibilidade
)

# Carregue os dados de validação
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse_categorical',
    subset='validation',  # Seleciona a parte de validação
    seed=123
)

# Construção do modelo (o mesmo modelo MNIST)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Camada convolucional
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilação do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo usando os geradores de dados
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Número de batches por época
    epochs=10,  # Aumente o número de épocas conforme necessário
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Salvar o modelo treinado
model.save('meu_modelo_digitos.keras')
print("Modelo treinado e salvo como 'meu_modelo_digitos.keras'")
