###########################################################################
###     R E C O N H E C E D O R    D E    D I G I T O S    M N I S T    ###
###########################################################################
### utiliza o modelo informado para realizar o reconhecimento do        ###
### digito existente na imagem PNG ou BMP                               ###
###########################################################################
### Prof. Filipo Mor - filipomor.com - github.com/ProfessorFilipo       ###
###########################################################################
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Solicita o nome do arquivo do modelo
nome_modelo = input("Digite o nome do arquivo do modelo salvo (.h5 ou .keras): ")

# Carrega o modelo
try:
    modelo = load_model(nome_modelo)
    print(f"Modelo '{nome_modelo}' carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# Solicita o nome da imagem de entrada
imagem_path = input("Digite o caminho da imagem de entrada (BMP ou PNG): ")

# Carrega a imagem, converte para escala de cinza e redimensiona para 28x28
try:
    img = Image.open(imagem_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    # Expande dimensões para o formato esperado pelo modelo (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)  # batch size 1
    img_array = np.expand_dims(img_array, axis=-1)  # canal
except Exception as e:
    print(f"Erro ao processar a imagem: {e}")
    exit()

# Faz a previsão
predicoes = modelo.predict(img_array)
digito_previsto = np.argmax(predicoes)

print(f"Previsão do dígito: {digito_previsto}")
print(f"Confiança: {np.max(predicoes)*100:.2f}%")
