import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Carregar o modelo treinado
model = load_model('mnist_model001.h5')

# Selecionar uma imagem aleatória do dataset MNIST de treino
# Consulte o dataset carregado no seu código original
# Assumindo que já tenha as variáveis train_images e train_labels disponíveis
# Caso contrário, carregue novamente ou adapte o código
from tensorflow.keras.datasets import mnist
(train_images, train_labels), _ = mnist.load_data()

# Seleciona um índice aleatório
idx = np.random.randint(0, len(train_images))
img_array = train_images[idx]

# Exibir a imagem original
plt.imshow(img_array, cmap='gray')
plt.title(f'Imagem original - label: {train_labels[idx]}')
plt.axis('off')
plt.show()

# Pré-processar a imagem para reconhecimento
img = Image.fromarray(img_array)
# Normalizar
img_norm = np.array(img) / 255.0
# Expandir dimensões para entrada do modelo
img_input = np.expand_dims(img_norm, axis=0)

# Fazer previsão
predictions = model.predict(img_input)
previsao = np.argmax(predictions)

print(f'Previsão do modelo para o dígito: {previsao}')

