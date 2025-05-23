from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Carregar o modelo (pode demorar um pouco na primeira vez)
print("Carregando o modelo...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Exemplo de sentenças para comparar
sentences = [
    "O gato está sentado no tapete",
    "Um felino está sobre o carpete",
    "O cachorro está brincando no parque",
    "O computador está processando dados",
    "O laptop está executando programas"
]

# Gerar embeddings
print("\nGerando embeddings...")
embeddings = model.encode(sentences)

# Calcular similaridade entre todas as sentenças
print("\nCalculando similaridades...")
similarities = cosine_similarity(embeddings)

# Mostrar resultados
print("\nSimilaridades entre as sentenças:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        print(f"\nSentença 1: {sentences[i]}")
        print(f"Sentença 2: {sentences[j]}")
        print(f"Similaridade: {similarities[i][j]:.4f}")

# Exemplo de busca semântica
print("\n\nExemplo de busca semântica:")
query = "animal de estimação"
query_embedding = model.encode([query])[0]

# Calcular similaridade da query com todas as sentenças
query_similarities = cosine_similarity([query_embedding], embeddings)[0]

# Ordenar resultados por similaridade
results = [(sentences[i], query_similarities[i]) for i in range(len(sentences))]
results.sort(key=lambda x: x[1], reverse=True)

print(f"\nResultados para a busca: '{query}'")
for sentence, score in results:
    print(f"Similaridade: {score:.4f} - {sentence}") 