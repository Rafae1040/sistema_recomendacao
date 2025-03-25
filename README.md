# 📚 Sistema de Recomendação de Livros

## 📖 Descrição do Projeto
Este projeto implementa um sistema de recomendação de livros utilizando **filtragem colaborativa**. O objetivo é sugerir livros para os usuários com base em suas preferências e no comportamento de outros leitores. A abordagem utilizada baseia-se no algoritmo **K-Nearest Neighbors (KNN)** para identificar livros semelhantes com base nas avaliações dos usuários.

## 🔗 Fonte de Dados
Os dados utilizados no projeto foram extraídos do **Kaggle**:
[Dataset Kaggle - Book Recommender System](https://www.kaggle.com/datasets/rxsraghavagrawal/book-recommender-system)

Os conjuntos de dados incluem:
- **📚 BX-Books.csv**: Informações sobre os livros (título, autor, ano de publicação, editora).
- **👤 BX-Users.csv**: Informações dos usuários (localização, idade).
- **⭐ BX-Book-Ratings.csv**: Avaliações de usuários sobre os livros.

## 🛠 Bibliotecas Utilizadas
O projeto faz uso das seguintes bibliotecas do Python:

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
```

## 🔍 Etapas do Projeto

### 1️⃣ Importação e Pré-processamento dos Dados
Os dados são carregados e pré-processados, incluindo:
- 🗑 Remoção de colunas desnecessárias.
- ✍️ Renomeação de colunas para facilitar a manipulação.
- 📊 Filtragem de livros com um mínimo de 50 avaliações.
- 🔄 Remoção de duplicatas de avaliações de um mesmo usuário para o mesmo livro.

```python
# 📥 Importação dos Dados
books = pd.read_csv('BX-Books.csv', sep=';', encoding="latin-1", on_bad_lines='skip')
users = pd.read_csv('BX-Users.csv', sep=';', encoding="latin-1", on_bad_lines='skip')
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

# 🏷 Renomeação de colunas
books = books [['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)
```

### 2️⃣ Criação da Matriz de Avaliações
A matriz de avaliações dos livros é criada, transpondo usuários em colunas:

```python
# 🔄 Criação da matriz de ratings
book_pivot = ratings.pivot_table(columns='user_id', index='title', values="rating")
book_pivot.fillna(0, inplace=True)

# 🔢 Convertendo para matriz esparsa
book_sparse = csr_matrix(book_pivot)
```

### 3️⃣ Treinamento do Modelo de Recomendação
O modelo KNN é treinado para encontrar livros similares com base nas avaliações dos usuários:

```python
# 🤖 Treinamento do modelo
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)
```

### 4️⃣ Geração de Recomendações
Para recomendar livros similares a um livro específico, utilizamos:

```python
# 📌 Obter recomendações para um livro específico
book_index = 0  # Exemplo: primeiro livro da lista

distances, suggestions = model.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1))

for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])
```

## 🚀 Aplicações e Benefícios
- 🎯 Personalização da experiência do usuário.
- 💰 Aumento nas taxas de conversão e faturamento.
- 🔗 Fortalecimento da relação com os clientes.
- 📈 Maior precisão nas recomendações com uso da distância euclidiana.

## 🎯 Conclusão
Ao implementar um sistema de recomendação eficaz, a empresa aumenta significativamente as taxas de conversão, oferecendo sugestões personalizadas que realmente interessam aos clientes. Isso impulsiona as vendas e contribui para o crescimento do faturamento, já que mais consumidores são atraídos por ofertas relevantes. Além disso, ao demonstrar um profundo entendimento das preferências do cliente, a empresa cria uma experiência de compra envolvente, resultando em maior satisfação e fidelização.

Para tornar esse processo mais preciso, muitas dessas recomendações utilizam a distância euclidiana, uma métrica que mede a similaridade entre usuários ou produtos. Em machine learning, essa técnica é usada para calcular a proximidade entre pontos, como no algoritmo KNN (K-Nearest Neighbors). Na computação gráfica, ela também ajuda a determinar distâncias entre objetos no espaço. Quanto menor a distância euclidiana entre dois pontos, maior a semelhança entre eles, permitindo que as recomendações sejam cada vez mais precisas e personalizadas.

Este sistema de recomendação de livros demonstra como técnicas de aprendizado de máquina podem ser aplicadas para melhorar a experiência do usuário e gerar valor para negócios no setor de livros e e-commerce. Além disso, esse conceito pode ser aplicado em diversas áreas, como recomendação de filmes, produtos em lojas virtuais, cursos online e até serviços médicos personalizados.

