# Importar as bibliotecas necessárias
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Inicializar SparkSession
spark = SparkSession.builder.appName("SparkCNNTraining").getOrCreate()

# Carregar o conjunto de dados CSV no Spark
start_time = time.time()
data_path = '/content/data.csv'
data = spark.read.csv(data_path, header=True, inferSchema=True)
load_time = time.time() - start_time
print(f"Tempo de carregamento dos dados: {load_time:.2f} segundos")

# Exibir o esquema do DataFrame para verificar os tipos de colunas
data.printSchema()

# 1. Remover a coluna '_c32' e 'id' que não são necessárias
data = data.drop('_c32', 'id')

# 2. Converter 'diagnosis' para binário: M = 1 (Maligno), B = 0 (Benigno)
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
data = indexer.fit(data).transform(data)

# 3. Combinar todas as features num único vetor
# Listar todas as colunas de features, exceto a 'diagnosis' e a nova coluna 'label'
feature_columns = data.columns
feature_columns.remove('diagnosis')
feature_columns.remove('label')

# Montar as features em um único vetor
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select('features', 'label')

# Dividir os dados em treino e teste (80% treino, 20% teste)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Normalizar os dados (StandardScaler)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select('scaledFeatures', 'label')
test_data = scaler_model.transform(test_data).select('scaledFeatures', 'label')

# 2. Converter para arrays NumPy para uso no TensorFlow
def spark_to_numpy(df):
    """Converte um Spark DataFrame para NumPy arrays."""
    features = np.array(df.select('scaledFeatures').rdd.map(lambda x: x[0]).collect())
    labels = np.array(df.select('label').rdd.map(lambda x: x[0]).collect())
    return features, labels

# Converter conjuntos de treino e teste para NumPy
X_train, y_train = spark_to_numpy(train_data)
X_test, y_test = spark_to_numpy(test_data)

# 3. Criar e treinar o modelo com TensorFlow
# Hiperparâmetros ajustáveis
dropout_rate = 0.4         # Dropout de 40%
l2_lambda = 0.01           # Regularização L2 ajustada

# Construir o modelo
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(l2_lambda)),
    layers.Dropout(dropout_rate),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
    layers.Dropout(dropout_rate),
    layers.Dense(1, activation='sigmoid')  # Saída para classificação binária
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks para Early Stopping e redução da taxa de aprendizado
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Treinar o modelo
start_time = time.time()
history = model.fit(
    X_train,          # Dados de treino
    y_train,          # Rótulos de treino
    epochs=150,       # Número máximo de épocas
    batch_size=32,    # Tamanho do lote
    validation_data=(X_test, y_test),  # Dados de validação
    callbacks=[early_stopping, reduce_lr]  # Callbacks
)
train_time = time.time() - start_time
print(f"Tempo de treinamento do modelo: {train_time:.2f} segundos")

# Visualizar a evolução do treino
# Plotar a acurácia
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia de Treinamento e Validação durante o Ajuste do Modelo CNN')
plt.xlabel('Número de Épocas')
plt.ylabel('Acurácia (%)')
plt.ylim([0.7, 1.0])  # Ajustar o eixo Y para focar na acurácia relevante
plt.legend(loc='lower right')

# Salvar ou mostrar o gráfico
plt.show()

# Plotar a perda
plt.plot(history.history['loss'], label='Perda de Treino')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
plt.show()
