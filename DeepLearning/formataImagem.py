from PIL import Image
import os


def preparar_imagem(input_path, output_path):
    """
    Carrega uma imagem BMP, converte para escala de cinza,
    redimensiona para 28x28 pixels e salva no caminho de saída.
    """
    # Abrir a imagem de entrada
    img = Image.open(input_path)

    # Converter para escala de cinza
    img = img.convert('L')

    # Redimensionar para 28x28
    img = img.resize((28, 28))

    # Salvar nova imagem BMP no caminho de saída
    img.save(output_path)
    print(f"Imagem processada salva em: {output_path}")


# Exemplo de uso:
# Substitua pelos seus caminhos
input_bmp = '04.bmp'
output_bmp = '04proc.bmp'

# Executar a função
preparar_imagem(input_bmp, output_bmp)

