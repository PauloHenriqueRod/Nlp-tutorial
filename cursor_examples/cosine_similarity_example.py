import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Exemplo 1: Vetores simples
print("Exemplo 1: Vetores simples")
v1 = np.array([1, 0, 0])  # Vetor apontando para x
v2 = np.array([0, 1, 0])  # Vetor apontando para y
v3 = np.array([1, 1, 0])  # Vetor apontando para x+y

# Calcular similaridades
sim_1_2 = cosine_similarity([v1], [v2])[0][0]  # Deve ser 0 (perpendiculares)
sim_1_3 = cosine_similarity([v1], [v3])[0][0]  # Deve ser ~0.707 (45 graus)
sim_2_3 = cosine_similarity([v2], [v3])[0][0]  # Deve ser ~0.707 (45 graus)

print(f"Similaridade entre v1 e v2: {sim_1_2:.3f}")
print(f"Similaridade entre v1 e v3: {sim_1_3:.3f}")
print(f"Similaridade entre v2 e v3: {sim_2_3:.3f}")

# Exemplo 2: Vetores de palavras
print("\nExemplo 2: Vetores de palavras (simplificado)")
# Vetores simplificados para demonstrar o conceito
gato = np.array([0.8, 0.2, 0.1])      # [animal, doméstico, felino]
cachorro = np.array([0.7, 0.3, 0.0])  # [animal, doméstico, canino]
carro = np.array([0.1, 0.0, 0.9])     # [veículo, motor, transporte]

# Calcular similaridades
sim_gato_cachorro = cosine_similarity([gato], [cachorro])[0][0]
sim_gato_carro = cosine_similarity([gato], [carro])[0][0]

print(f"Similaridade entre 'gato' e 'cachorro': {sim_gato_cachorro:.3f}")
print(f"Similaridade entre 'gato' e 'carro': {sim_gato_carro:.3f}")

# Exemplo 3: Visualização da fórmula
print("\nExemplo 3: Cálculo manual da similaridade do cosseno")
def manual_cosine_similarity(v1, v2):
    # Produto escalar
    dot_product = np.dot(v1, v2)
    # Normas (magnitudes)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Similaridade do cosseno
    return dot_product / (norm_v1 * norm_v2)

# Testar com os vetores do exemplo 1
manual_sim_1_2 = manual_cosine_similarity(v1, v2)
print(f"Similaridade manual entre v1 e v2: {manual_sim_1_2:.3f}")
print(f"Similaridade scikit-learn entre v1 e v2: {sim_1_2:.3f}") 