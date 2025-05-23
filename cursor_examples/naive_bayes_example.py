from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Exemplo de dados de treinamento (classificação de sentimentos)
textos = [
    "Adorei o produto, muito bom!",
    "Excelente atendimento, recomendo",
    "Produto chegou antes do prazo",
    "Não gostei, qualidade ruim",
    "Péssimo serviço, não recomendo",
    "Produto com defeito, decepcionante",
    "Ótima experiência de compra",
    "Muito satisfeito com a compra",
    "Produto não atendeu minhas expectativas",
    "Serviço horrível, nunca mais compro",
    "O produto é muito bom, recomendo",
    "O produto é muito ruim, não recomendo",
    "Atendimaneto muito demorado, desisti da compra",
    "Entrega rápida, muito satisfeito",
    "Produto não atendeu minhas expectativas"
]

# Labels: 1 para positivo, 0 para negativo
labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    textos, labels, test_size=0.3, random_state=42
)

# Vetorização dos textos
print("Vetorizando os textos...")
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinar o modelo
print("\nTreinando o modelo Naive Bayes...")
modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)

# Fazer predições
print("\nFazendo predições...")
predicoes = modelo.predict(X_test_vec)

# Avaliar o modelo
print("\nRelatório de Classificação:")
print(classification_report(y_test, predicoes, target_names=['Negativo', 'Positivo']))

# Matriz de confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, predicoes))

# Exemplo de predição para novos textos
novos_textos = [
    "Produto muito bom, recomendo",
    "Não gostei nada do produto",
    "Atendimento excelente, mas o produto é mediano",
    "Entrega rápida, a compra valeu a pena",
    "Produto não atendeu minhas expectativas, mas o atendimento foi bom",
    "Entrega rápida mas o produto veio com defeitos"
]

print("\nPredições para novos textos:")
novos_textos_vec = vectorizer.transform(novos_textos)
novas_predicoes = modelo.predict(novos_textos_vec)
probabilidades = modelo.predict_proba(novos_textos_vec)

for texto, pred, prob in zip(novos_textos, novas_predicoes, probabilidades):
    sentimento = "Positivo" if pred == 1 else "Negativo"
    print(f"\nTexto: {texto}")
    print(f"Sentimento: {sentimento}")
    print(f"Probabilidade: {prob[pred]:.2%}")

# Mostrar as palavras mais importantes para cada classe
print("\nPalavras mais importantes para cada classe:")
feature_names = vectorizer.get_feature_names_out()
for i, classe in enumerate(['Negativo', 'Positivo']):
    # Pegar os coeficientes do modelo para esta classe
    coefs = modelo.feature_log_prob_[i]
    # Pegar os índices das 5 palavras mais importantes
    top_indices = np.argsort(coefs)[-5:]
    print(f"\n{classe}:")
    for idx in top_indices:
        print(f"- {feature_names[idx]}: {coefs[idx]:.3f}") 