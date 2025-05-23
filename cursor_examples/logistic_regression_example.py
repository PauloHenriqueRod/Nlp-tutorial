from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

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
    "Atendimento muito demorado, desisti da compra",
    "Entrega rápida, muito satisfeito",
    "Produto não atendeu minhas expectativas",
    "Qualidade excelente, superou expectativas",
    "Preço muito alto para o que oferece",
    "Bom custo-benefício",
    "Não vale o preço cobrado",
    "Recomendo fortemente a todos"
]

# Labels: 1 para positivo, 0 para negativo
labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    textos, labels, test_size=0.3, random_state=42
)

# Vetorização dos textos usando TF-IDF
print("Vetorizando os textos...")
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinar o modelo com regularização L2
print("\nTreinando o modelo de Regressão Logística...")
modelo = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)
modelo.fit(X_train_vec, y_train)

# Validação cruzada
print("\nRealizando validação cruzada...")
scores = cross_val_score(modelo, X_train_vec, y_train, cv=5)
print(f"Acurácia média na validação cruzada: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Fazer predições
print("\nFazendo predições...")
predicoes = modelo.predict(X_test_vec)
probabilidades = modelo.predict_proba(X_test_vec)

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
novas_probabilidades = modelo.predict_proba(novos_textos_vec)

for texto, pred, prob in zip(novos_textos, novas_predicoes, novas_probabilidades):
    sentimento = "Positivo" if pred == 1 else "Negativo"
    print(f"\nTexto: {texto}")
    print(f"Sentimento: {sentimento}")
    print(f"Probabilidade: {prob[pred]:.2%}")

# Mostrar as palavras mais importantes para cada classe
print("\nPalavras mais importantes para cada classe:")
feature_names = vectorizer.get_feature_names_out()
for i, classe in enumerate(['Negativo', 'Positivo']):
    # Pegar os coeficientes do modelo para esta classe
    coefs = modelo.coef_[0]
    # Pegar os índices das 5 palavras mais importantes
    if classe == 'Positivo':
        top_indices = np.argsort(coefs)[-5:]
    else:
        top_indices = np.argsort(coefs)[:5]
    print(f"\n{classe}:")
    for idx in top_indices:
        print(f"- {feature_names[idx]}: {coefs[idx]:.3f}")

# Plotar curva ROC
fpr, tpr, _ = roc_curve(y_test, probabilidades[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show() 