# Importar as bibliotecas necessárias
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from tensorflow.keras import models, layers, regularizers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Inicializar SparkSession
spark = SparkSession.builder.appName("SparkResNetTraining").getOrCreate()

# Carregar o conjunto de dados CSV no Spark
start_time = time.time()
data_path = 'data.csv'
data = spark.read.csv(data_path, header=True, inferSchema=True)
load_time = time.time() - start_time
print(f"Tempo de carregamento dos dados: {load_time:.2f} segundos")

# Remover colunas irrelevantes
data = data.drop('_c32', 'id')

# Converter 'diagnosis' para binário
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
data = indexer.fit(data).transform(data)

# Combinar features em um vetor
feature_columns = [col for col in data.columns if col not in ('diagnosis', 'label')]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select('features', 'label')

# Dividir e normalizar os dados
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select('scaledFeatures', 'label')
test_data = scaler_model.transform(test_data).select('scaledFeatures', 'label')

# Converter para arrays NumPy e redimensionar
def spark_to_numpy(df):
    features = np.array(df.select('scaledFeatures').rdd.map(lambda x: x[0]).collect())
    labels = np.array(df.select('label').rdd.map(lambda x: x[0]).collect())
    return features, labels

X_train, y_train = spark_to_numpy(train_data)
X_test, y_test = spark_to_numpy(test_data)

# Redimensionar para formato 3D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Bloco Residual com Regularização
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    # Primeira camada convolucional com L2
    x = layers.Conv1D(
        filters, kernel_size,
        strides=stride,
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)  # Dropout adicionado
    
    # Segunda camada convolucional com L2
    x = layers.Conv1D(
        filters, kernel_size,
        strides=1,
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Atalho com regularização
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters, 1,
            strides=stride,
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Construir ResNet Adaptada
input_tensor = Input(shape=(X_train.shape[1], 1))
x = layers.Conv1D(64, 7, strides=2, padding='same')(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

# Sequência de Blocos Residuals
x = residual_block(x, 64, stride=1)
x = residual_block(x, 64, stride=1)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 128, stride=1)
x = residual_block(x, 256, stride=2)
x = residual_block(x, 256, stride=1)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(
    256,  # Reduzido de 512
    activation='relu',
    kernel_regularizer=regularizers.l2(0.01)
)(x)
x = layers.Dropout(0.5)(x)  # Aumentado de 0.4
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=output)

# Compilar o modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks ajustados
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,  # Aumentado
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,  # Reduzido
    patience=10,  # Aumentado
    min_lr=1e-6
)

# Treinar o modelo
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)
print(f"Tempo de treinamento: {time.time() - start_time:.2f} segundos")

# Plotar métricas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda')
plt.legend()
plt.show()

# Encerrar Spark
spark.stop()
