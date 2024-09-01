# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# from sklearn.metrics import classification_report
# import numpy as np

# # Directorios de datos
# train_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\entrenamiento'
# validation_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\validacion'

# # Configuración de generadores de datos
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# validation_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )

# # Cargar la ResNet-50 preentrenada sin las capas superiores (head)
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Añadir capas superiores personalizadas
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.5)(x)  # Añadir dropout
# predictions = Dense(1, activation='sigmoid')(x)

# # Crear el modelo completo
# model = Model(inputs=base_model.input, outputs=predictions)

# # Descongelar las últimas capas
# for layer in base_model.layers[-10:]:
#     layer.trainable = True

# # Compilar el modelo con la tasa de aprendizaje ajustada
# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# # Configuración de la política de reducción de la tasa de aprendizaje
# lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-6)

# # Entrenamiento del modelo
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=50,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     callbacks=[lr_reduction]
# )

# # Evaluación en el conjunto de validación
# loss, accuracy = model.evaluate(validation_generator)
# print(f'Loss en validación: {loss}')
# print(f'Accuracy en validación: {accuracy}')

# # Matriz de confusión
# validation_generator.reset()
# Y_pred = model.predict(validation_generator)
# y_pred = np.round(Y_pred).astype(int)
# y_true = validation_generator.classes

# cm = confusion_matrix(y_true, y_pred)
# print('Matriz de confusión:')
# print(cm)

# # Reporte de clasificación
# print(classification_report(y_true, y_pred))

# # Guardar el modelo entrenado
# model.save('resnet50_capacidad_amortiguacion.h5')

# print("Entrenamiento completado, evaluación realizada, y modelo guardado.")

#-------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import numpy as np

# Directorios de datos (ajusta las rutas a tus carpetas)
train_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\entrenamiento'
validation_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\validacion'

# Configuración de generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,#0.2,
    height_shift_range=0.1,#0.2,
    shear_range=0.1,#0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Cargar la ResNet-50 preentrenada sin las capas superiores (head)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Añadir capas superiores personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Crear el modelo completo
model = Model(inputs=base_model.input, outputs=predictions)

# Descongelar las últimas 10 capas de la base del modelo (ResNet-50)
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compilar el modelo con el optimizador Adam y la tasa de aprendizaje especificada
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluación en el conjunto de validación
loss, accuracy = model.evaluate(validation_generator)
print(f'Loss en validación: {loss}')
print(f'Accuracy en validación: {accuracy}')

# Matriz de confusión
validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = np.round(Y_pred).astype(int)
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
print('Matriz de confusión:')
print(cm)

# Guardar el modelo entrenado
model.save('resnet50_capacidad_amortiguacion.h5')

print("Entrenamiento completado, evaluación realizada, y modelo guardado.")

#-------------------------------------------------------------------------------------------------------------------------------------
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam

# # Directorios de datos (ajusta las rutas a tus carpetas)
# train_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\entrenamiento'
# validation_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\validacion'

# # Configuración de generadores de datos
# train_datagen = ImageDataGenerator(
#     rescale=1./255,            # Normaliza las imágenes al rango [0, 1]
#     rotation_range=40,         # Rango de rotación aleatoria en grados
#     width_shift_range=0.2,     # Desplazamiento horizontal aleatorio
#     height_shift_range=0.2,    # Desplazamiento vertical aleatorio
#     shear_range=0.2,           # Transformación de corte aleatoria
#     zoom_range=0.2,            # Zoom aleatorio
#     horizontal_flip=True,      # Volteo horizontal aleatorio
#     fill_mode='nearest'        # Estrategia de relleno para los píxeles recién creados
# )

# validation_datagen = ImageDataGenerator(rescale=1./255)  # Solo normaliza las imágenes al rango [0, 1] para el conjunto de validación

# # Generadores de datos para el entrenamiento y la validación
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),  # Redimensiona las imágenes a 224x224 píxeles
#     batch_size=32,           # Tamaño del lote
#     class_mode='binary'      # Clasificación binaria
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(224, 224),  # Redimensiona las imágenes a 224x224 píxeles
#     batch_size=32,           # Tamaño del lote
#     class_mode='binary'      # Clasificación binaria
# )

# # Cargar la ResNet-50 preentrenada sin las capas superiores (head)
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Añadir capas superiores personalizadas
# x = base_model.output
# x = GlobalAveragePooling2D()(x)   # Capa de pooling global promedio
# x = Dense(1024, activation='relu')(x)  # Capa densa con 1024 unidades y activación ReLU
# predictions = Dense(1, activation='sigmoid')(x)  # Capa de salida con una unidad y activación sigmoide (para clasificación binaria)

# # Crear el modelo completo
# model = Model(inputs=base_model.input, outputs=predictions)

# # Descongelar las últimas 10 capas de la base del modelo (ResNet-50)
# for layer in base_model.layers[-10:]:
#     layer.trainable = True

# # Compilar el modelo con el optimizador Adam y la tasa de aprendizaje especificada
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# # Entrenar el modelo con más épocas
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Número de pasos por época
#     epochs=30,  # Aumentar a 30 épocas
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size  # Número de pasos de validación por época
# )

# # Guardar el modelo entrenado
# model.save('resnet50_capacidad_amortiguacion.h5')

# print("Entrenamiento completado y modelo guardado.")

#---------------------------------------------------------------------------------------------------------------------------------------
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam

# # Directorios de datos (ajusta las rutas a tus carpetas)
# train_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\entrenamiento'
# validation_dir = r'C:\Users\ezean\OneDrive\Escritorio\TESIS ANTRAYGUES\prueba resnet\imagenes\capacidad de amortiguacion\validacion'

# # Configuración de generadores de datos
# train_datagen = ImageDataGenerator(
#     rescale=1./255,            # Normaliza las imágenes al rango [0, 1]
#     rotation_range=40,         # Rango de rotación aleatoria en grados
#     width_shift_range=0.2,     # Desplazamiento horizontal aleatorio
#     height_shift_range=0.2,    # Desplazamiento vertical aleatorio
#     shear_range=0.2,           # Transformación de corte aleatoria
#     zoom_range=0.2,            # Zoom aleatorio
#     horizontal_flip=True,      # Volteo horizontal aleatorio
#     fill_mode='nearest'        # Estrategia de relleno para los píxeles recién creados
# )

# validation_datagen = ImageDataGenerator(rescale=1./255)  # Solo normaliza las imágenes al rango [0, 1] para el conjunto de validación

# # Generadores de datos para el entrenamiento y la validación
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),  # Redimensiona las imágenes a 224x224 píxeles
#     batch_size=32,           # Tamaño del lote
#     class_mode='binary'      # Clasificación binaria
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(224, 224),  # Redimensiona las imágenes a 224x224 píxeles
#     batch_size=32,           # Tamaño del lote
#     class_mode='binary'      # Clasificación binaria
# )

# # Cargar la ResNet-50 preentrenada sin las capas superiores (head)
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Añadir capas superiores personalizadas
# x = base_model.output
# x = GlobalAveragePooling2D()(x)   # Capa de pooling global promedio
# x = Dense(1024, activation='relu')(x)  # Capa densa con 1024 unidades y activación ReLU
# predictions = Dense(1, activation='sigmoid')(x)  # Capa de salida con una unidad y activación sigmoide (para clasificación binaria)

# # Crear el modelo completo
# model = Model(inputs=base_model.input, outputs=predictions)

# # Congelar las capas de la base del modelo (ResNet-50)
# for layer in base_model.layers:
#     layer.trainable = False

# # Compilar el modelo con el optimizador Adam y la tasa de aprendizaje especificada
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# # Entrenar el modelo
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Número de pasos por época
#     epochs=10,  # Número de épocas
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size  # Número de pasos de validación por época
# )

# # Guardar el modelo entrenado
# model.save('resnet50_capacidad_amortiguacion.h5')

# print("Entrenamiento completado y modelo guardado.")


