# Reconhecimento de dígitos

TODO replace image by a diagram
![Arquitetura](./docs/arc.png)


## Aplicação (TODO)

### Módulo `classification engine`
Módulo que contém os códigos utilizados durante a investigação e modelagem do modelo de classificação, o modelo pré-treinado está no arquivo `/digits-classification/classification-engine/model_v1.h5`.

#### Modelo de classificação
O modelo utilizado na classificação dos dígitos é um `MLP` com duas camadas ocultas que segue a seguinte arquitetura.

| Camada    | Parâmetros                                                         |
|-----------|--------------------------------------------------------------------|
| Entrada   | input_shape=(28,28)                                                |
| Dense (1) | units=256, activation='relu', kernel_initializer=RandomNormal()    |
| Dense (2) | units=128, activation='sigmoid', kernel_initializer=RandomNormal() |
| Saída     | units=10, activation='softmax'

Os hiperparâmetros utilizados foram:
  - `epochs`: 50;
  - `optimizer`: `rmsprop`;
  - `loss`: categorical_crossentropy;
  - `validation_split`: 0.2.

#### Resultados

| Base      | Acurácia | Perda |
|-----------|----------|-------|
| teste     | 0.979    | 0.124 |
| validação | 0.989    | 0.06  |

## Classification API

Api desenvolvida em Flask que retorna a classe e as probabilidades para um conjunto de valores.

### Rotas

  - POST classify_digit
    - recebe parâmetro image contendo um array com 4096 posições com os valores de cada um dos pixels de uma imagem de dígito e retorna a classe e o conjunto de probabilidades para cada uma das classes possíveis;
    - exemplo de response:
    ```JSON
    {
      "digit": 0,
      "digits_probabilities": [
        0.9999998807907104,
        1.6444052066202919e-12,
        7.89274778867366e-08,
        5.124025093117268e-10,
        1.272474625668707e-10,
        1.7181628564344464e-08,
        1.4171525286599262e-08,
        4.623578853113486e-09,
        2.1767440280817674e-11,
        4.111644802407e-09
      ]
    }
    ```

### TODO:
  - adicionar cobertura de testes;
  - adicionar Readme.MD ao módulo da api;
  - validar erro relacionado ao estouro de momória relativo ao keras.

## Setup (TODO)
  - `classification engine`:

## Notes:

## TODO:
  - [ ] Add readme classification-engine;
  - [ ] Describe scripts on classification engine module;
  - [ ] Add flask api that loads pre-trained model and return a image class.
