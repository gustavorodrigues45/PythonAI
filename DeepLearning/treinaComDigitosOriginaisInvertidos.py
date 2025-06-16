###########################################################################
###   T R E I N A D O R   D E  M O D E L O   D I G I T O S   M N I S T  ###
###########################################################################
### treina o modelo com as imagens MNIST e tambémm com as mesmas imagens###
### com cores invertidas, geradas em uma pasta solicitada ao usuário    ###
###########################################################################
### Prof. Filipo Mor - filipomor.com - github.com/ProfessorFilipo       ###
###########################################################################

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Função para carregar imagens BMP de uma pasta específica, filtrando por dígitos
def carregar_imagens(pasta, digits):
    images = []
    labels = []
    for filename in os.listdir(pasta):
        if filename.endswith('.bmp'):
            parts = filename.split('_')
            digit_part = parts[0]
            if digit_part.isdigit() and int(digit_part) in digits:
                filepath = os.path.join(pasta, filename)
                img = Image.open(filepath).convert('L')
                images.append(np.array(img))
                labels.append(int(digit_part))
    return np.array(images), np.array(labels)

# Entrada do usuário para o intervalo dos dígitos
start_digit = int(input("Informe o dígito inicial (exemplo: 0): "))
end_digit = int(input("Informe o dígito final (exemplo: 3): "))

digits_to_use = list(range(start_digit, end_digit + 1))
print(f"Treinando com os dígitos: {digits_to_use}")

# Carregar imagens originais do MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Filtrar pelos dígitos selecionados
train_filter = (y_train >= start_digit) & (y_train <= end_digit)
test_filter = (y_test >= start_digit) & (y_test <= end_digit)

x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

# Carregar imagens invertidas
folder_inverted = 'mnist_bmp_inverted'  # ajuste esse caminho, se necessário
x_inverted_train, y_inverted_train = carregar_imagens(folder_inverted, digits_to_use)
x_inverted_test, y_inverted_test = carregar_imagens(folder_inverted, digits_to_use)

# Combinar todos os dados
x_train_total = np.concatenate((x_train_filtered, x_inverted_train), axis=0)
y_train_total = np.concatenate((y_train_filtered, y_inverted_train), axis=0)
x_test_total = np.concatenate((x_test_filtered, x_inverted_test), axis=0)
y_test_total = np.concatenate((y_test_filtered, y_inverted_test), axis=0)

# Normalizar
x_train_total = x_train_total.astype('float32') / 255.
x_test_total = x_test_total.astype('float32') / 255.

# Expandir dimensões para CNN
x_train_total = np.expand_dims(x_train_total, -1)
x_test_total = np.expand_dims(x_test_total, -1)

# One-hot encoding das labels
y_train_categorical = to_categorical(y_train_total, 10)
y_test_categorical = to_categorical(y_test_total, 10)

# Construção do modelo
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train_total, y_train_categorical, epochs=10, batch_size=128,
          validation_data=(x_test_total, y_test_categorical))

# Solicitar ao usuário o nome para salvar o modelo
nome_modelo = input("Digite o nome para salvar o modelo (sem extensão): ")

# Salvar em formato H5
arquivo_h5 = f"{nome_modelo}.h5"
model.save(arquivo_h5)
print(f"Modelo salvo em {arquivo_h5}")

# Salvar em formato Keras (.keras)
arquivo_keras = f"{nome_modelo}.keras"
model.save(arquivo_keras)
print(f"Modelo salvo em {arquivo_keras}")

print("Processo concluído!")
