###########################################################################
###   I N V E R S O R    D E   C O R E S    I M A G E N S    M N I S T  ###
###########################################################################
### inverte as cores das imagens do MNIST em uma pasta                  ###
###########################################################################
### Prof. Filipo Mor - filipomor.com - github.com/ProfessorFilipo       ###
###########################################################################

import os
from PIL import Image

# Diretório onde as imagens estão salvas
input_dir = 'mnist_bmp'
# Diretório onde as imagens invertidas serão salvas
output_dir = 'mnist_bmp_inverted'

os.makedirs(output_dir, exist_ok=True)

# Lista todos os arquivos BMP no diretório de entrada
for filename in os.listdir(input_dir):
    if filename.endswith('.bmp'):
        filepath = os.path.join(input_dir, filename)
        # Abre a imagem
        img = Image.open(filepath)
        # Inverte as cores
        inverted_img = Image.eval(img, lambda x: 255 - x)
        # Cria o novo nome com o sufixo "_inverted.bmp"
        name_part = filename[:-4]  # remove ".bmp"
        new_filename = f"{name_part}_inverted.bmp"
        new_filepath = os.path.join(output_dir, new_filename)
        # Salva a imagem invertida
        inverted_img.save(new_filepath)
        print(f"Invertida e salva: {new_filepath}")

print("Inversão de cores concluída!")
