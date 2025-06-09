######################################################
###            D E E P   L E A R N I N G           ###
######################################################
### Exemplo de Deep Learning utilizando MNIST      ###
### Prof. Filipo Mor - 02 de junho de 2025         ###
######################################################

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image  # Import PIL for image loading and resizing

# Carregar o modelo salvo
model = load_model('model001.keras')

def reconhecer_digito(image_path):
    """
    Reconhece um dígito em uma imagem BMP ou PNG.

    Args:
        image_path (str): O caminho para o arquivo da imagem.

    Returns:
        int: O dígito reconhecido (0-9) ou None se houver um erro.
    """
    try:
        # 1. Carregar a imagem usando PIL (pillow)
        img = Image.open(image_path).convert('L') # Convert to grayscale

        # 2. Redimensionar a imagem para 28x28 pixels
        img = img.resize((28, 28), Image.LANCZOS) # Use LANCZOS for best quality

        # 3. Converter para um array NumPy
        img_array = np.array(img)

        # teste: 3.5: inverte as cores!
        img_array = 255 - img_array

        # 4. Pré-processamento: normalizar a imagem para valores entre 0 e 1
        img_array = img_array / 255.0

        # 5. Adicionar uma dimensão para que a imagem tenha o formato (1, 28, 28)
        img_array = np.expand_dims(img_array, axis=0)

        # 6. Fazer a previsão com o modelo
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        return predicted_class

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None


# Exemplo de uso
if __name__ == "__main__":
    while True:
        image_path = input("Digite o caminho da imagem (BMP ou PNG, ou 'sair' para encerrar): ")
        if image_path.lower() == 'sair':
            break

        if not os.path.exists(image_path):
            print("Arquivo não encontrado.")
            continue

        digito_reconhecido = reconhecer_digito(image_path)

        if digito_reconhecido is not None:
            print(f"O dígito reconhecido na imagem é: {digito_reconhecido}")
