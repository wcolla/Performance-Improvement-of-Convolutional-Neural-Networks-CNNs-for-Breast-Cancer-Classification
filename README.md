CNN Residual Network para Classificação com Spark e TensorFlow

# Descrição

Este projeto implementa uma rede neural convolucional residual (ResNet) para classificação binária de dados utilizando TensorFlow/Keras e Apache Spark. O código processa um conjunto de dados CSV, normaliza os valores e treina um modelo utilizando uma ResNet adaptada para dados tabulares.

# Bibliotecas Utilizadas

time: Medir o tempo de execução das operações

numpy: Manipulação de arrays numéricos

pyspark.sql.SparkSession: Inicialização do Spark para manipulação de grandes volumes de dados

pyspark.ml.feature: Ferramentas para transformação e normalização dos dados

tensorflow.keras: Implementação da rede neural

matplotlib.pyplot: Visualização dos resultados

# Estrutura do Código

# 1. Inicialização do Spark

O Spark é iniciado para manipulação do conjunto de dados de forma distribuída.

# 2. Carregamento e Processamento dos Dados

O conjunto de dados é carregado a partir de um arquivo CSV.

Remoção de colunas irrelevantes.

Conversão da coluna de diagnóstico para um formato binário.

Vetorizamos os dados para transformação em um modelo numérico adequado.

Os dados são divididos em treino e teste.

É aplicada a normalização para padronizar os valores das features.

# 3. Conversão para NumPy

Os dados são convertidos do formato Spark para arrays NumPy, necessários para o treinamento no TensorFlow.

# 4. Construção da Rede Residual (ResNet)

A rede é composta por:

Camada convolucional inicial para extração de features.

Múltiplos blocos residuais com regularização L2 e dropout para evitar overfitting.

Camada de pooling global e camadas densas para classificação final.

# 5. Compilação e Treinamento do Modelo

O modelo é compilado com adam e função de perda binary_crossentropy.

Utiliza EarlyStopping e ReduceLROnPlateau para ajustar dinamicamente a taxa de aprendizado e evitar overfitting.

O modelo é treinado por até 150 épocas.

# 6. Avaliação e Visualização dos Resultados

O histórico de treinamento é utilizado para gerar gráficos de acurácia e perda.

# 7. Finalização

O Spark é encerrado após o processamento dos dados.

# Como Executar

Certifique-se de ter o Apache Spark, TensorFlow e dependências instaladas.

Altere o caminho do arquivo CSV conforme necessário.

Execute o script Python para processar os dados e treinar a rede neural.

# Observações

O Spark é utilizado para manipulação eficiente de grandes volumes de dados.

A arquitetura ResNet permite capturar padrões mais complexos nos dados.

Parâmetros como dropout, regularização e taxa de aprendizado foram ajustados para melhor desempenho.

# Autor

William Colla
