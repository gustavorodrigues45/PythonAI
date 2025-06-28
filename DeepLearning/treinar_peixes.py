import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset CIFAR-100
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# Mapeamento de algumas classes do CIFAR-100 para referência
# A classe 'fish' no CIFAR-100 tem o label 35.
# Vamos treinar o modelo para reconhecer peixes e alguns outros animais aquáticos e um inseto.
# Isso torna o desafio mais interessante do que apenas classificar "peixe" vs "não-peixe".
class_mapping = {
    35: 'fish',
    28: 'dolphin',
    83: 'seal',
    99: 'whale',
    7: 'bee'
}
target_labels = list(class_mapping.keys())
num_classes = len(target_labels)

# Mapear os labels originais para novos labels (0 a 4)
label_map = {original_label: new_label for new_label, original_label in enumerate(target_labels)}
reverse_label_map = {new_label: original_label for new_label, original_label in enumerate(target_labels)}

def filter_and_remap_labels(images, labels):
    """Filtra as imagens e os rótulos para as classes de interesse e remapeia os rótulos."""
    mask = np.isin(labels, target_labels).flatten()
    filtered_images = images[mask]
    filtered_labels_original = labels[mask]
    remapped_labels = np.array([label_map[l[0]] for l in filtered_labels_original])
    return filtered_images, remapped_labels

# Filtrar e remapear os dados de treino e teste
train_images_filtered, train_labels_filtered = filter_and_remap_labels(train_images, train_labels)
test_images_filtered, test_labels_filtered = filter_and_remap_labels(test_images, test_labels)

# Normalizar as imagens para valores entre 0 e 1
train_images_filtered = train_images_filtered / 255.0
test_images_filtered = test_images_filtered / 255.0

# Visualizar algumas imagens do novo dataset
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images_filtered[i], cmap=plt.cm.binary)
    # Pega o novo rótulo, converte para o rótulo original do CIFAR e depois para o nome da classe
    original_label = reverse_label_map[train_labels_filtered[i]]
    plt.xlabel(class_mapping[original_label])
plt.show()


# Construção do modelo de rede neural (similar ao ExemploMNIST.py mas adaptado para CIFAR-100)
# As imagens do CIFAR-100 são 32x32 com 3 canais de cor (RGB)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax')) # num_classes é o número de classes que definimos

model.summary()

# Compilação do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(train_images_filtered, train_labels_filtered, epochs=15,
                    validation_data=(test_images_filtered, test_labels_filtered))

# Salvar o modelo treinado no formato H5
model.save('cifar100_fish_model.h5')
print("Modelo treinado e salvo como 'cifar100_fish_model.h5'")

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_images_filtered,  test_labels_filtered, verbose=2)
print(f'\nAcurácia no teste: {test_acc:.4f}')

# Teste com uma imagem específica
print("\n--- Testando o reconhecimento com uma imagem ---")
# Pega uma imagem de peixe do conjunto de teste
fish_test_images = test_images_filtered[test_labels_filtered == label_map[35]]
if len(fish_test_images) > 0:
    test_image = fish_test_images[0]
    plt.imshow(test_image)
    plt.title("Imagem de teste (Peixe)")
    plt.show()

    # Prepara a imagem para o modelo
    img_array = np.expand_dims(test_image, axis=0)

    # Faz a previsão
    predictions = model.predict(img_array)
    predicted_label_index = np.argmax(predictions)
    predicted_class_name = class_mapping[reverse_label_map[predicted_label_index]]
    confidence = 100 * np.max(predictions)

    print(f"Previsão do modelo: '{predicted_class_name}' com {confidence:.2f}% de confiança.")
else:
    print("Não foram encontradas imagens de peixe no conjunto de teste filtrado para fazer um teste de reconhecimento.")