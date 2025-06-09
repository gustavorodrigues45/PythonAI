import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Supondo que seu modelo já esteja treinado e salvo
# Caso não, você pode salvar com: model.save('meu_model.h5')
# E carregar aqui:
model = load_model('mnist_model001.h5')

# Carregar sua imagem (substitua pelo caminho da sua imagem)
#img_path = 'caminho/para/sua_imagem.png'
img_path = '04.png'
img = Image.open(img_path).convert('L')  # converte para escala de cinza
img = img.resize((28, 28))  # redimensiona para 28x28

# Converter para array e normalizar
img_array = np.array(img) / 255.0
# Adicionar uma dimensão extra para compatibilidade com o modelo (batch size=1)
img_array = np.expand_dims(img_array, axis=0)

# Fazer a predição
predictions = model.predict(img_array)

# Obter o índice da classe com maior probabilidade
predicted_label = np.argmax(predictions)

print(f"Previsão do modelo: {predicted_label}")

# Opcional: mostrar a imagem carregada
plt.imshow(img, cmap='gray')
plt.title(f'Predito: {predicted_label}')
plt.axis('off')
plt.show()

