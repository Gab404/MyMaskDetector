import matplotlib.pyplot as plt

from preprocess import getDataset
from model import getModel

train_gen, val_gen = getDataset()

model = getModel()
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)

# Save le model
model.save("mask_detector_cnn.h5")
print("✅ Modèle entraîné et sauvegardé sous mask_detector_cnn.h5")

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Précision du modèle')
plt.xlabel('Épochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
