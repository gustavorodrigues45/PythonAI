import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Mapeamento de classes utilizado no treinamento
class_mapping = {
    0: 'fish',
    1: 'dolphin',
    2: 'seal',
    3: 'whale',
    4: 'bee'
}

# Carregar o modelo treinado
model = load_model('cifar100_fish_model.h5')

# Função para preparar a imagem
def preparar_imagem(img_path):
    img = Image.open(img_path).resize((32, 32))
    img_array = np.array(img)
    # Garante que a imagem tenha 3 canais de cor
    if img_array.ndim == 2: # Se for escala de cinza
        img_array = np.stack((img_array,)*3, axis=-1)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Caminho para a imagem que você quer classificar
# Crie uma pasta 'exemplos' e coloque uma imagem de peixe nela.
# Por exemplo: 'exemplos/meu_peixe.jpg'
caminho_imagem = 'exemplos/meu_peixe.jpg' # SUBSTITUA PELO CAMINHO DA SUA IMAGEM

try:
    # Preparar a imagem
    imagem_para_previsao = preparar_imagem(caminho_imagem)

    # Fazer a predição
    predictions = model.predict(imagem_para_previsao)
    predicted_label_index = np.argmax(predictions)
    predicted_class_name = class_mapping[predicted_label_index]
    confidence = 100 * np.max(predictions)

    # Mostrar a imagem e o resultado
    plt.imshow(Image.open(caminho_imagem))
    plt.title(f"Previsão: {predicted_class_name} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"O modelo previu que a imagem é um '{predicted_class_name}' com {confidence:.2f}% de confiança.")

except FileNotFoundError:
    print(f"Erro: O arquivo de imagem não foi encontrado em '{caminho_imagem}'.")
    print("Por favor, crie uma pasta chamada 'exemplos', coloque uma imagem nela e atualize a variável 'caminho_imagem' no código.")