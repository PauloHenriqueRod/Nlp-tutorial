import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def exemplo_sentence_encoder():
    print("\n=== EXEMPLO SENTENCE ENCODER ===")
    
    # Carregar modelo de Sentence Encoder
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Exemplos de sentenças
    sentencas = [
        "O gato está sentado no tapete",
        "Um felino está sobre o carpete",
        "O cachorro está brincando no jardim",
        "O computador está ligado na mesa"
    ]
    
    # Gerar embeddings
    embeddings = model.encode(sentencas)
    
    # Calcular similaridade entre todas as sentenças
    print("\nSimilaridade entre sentenças:")
    for i in range(len(sentencas)):
        for j in range(i+1, len(sentencas)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"\n'{sentencas[i]}' vs '{sentencas[j]}'")
            print(f"Similaridade: {sim:.3f}")
            
            # Interpretação
            if sim > 0.7:
                print("→ Muito similares (mesmo significado)")
            elif sim > 0.5:
                print("→ Moderadamente similares")
            else:
                print("→ Pouco similares (significados diferentes)")

def exemplo_classificador():
    print("\n=== EXEMPLO CLASSIFICADOR ===")
    
    # Dados de treino
    textos = [
        "O gato está sentado no tapete",
        "O cachorro está brincando no jardim",
        "O computador está ligado na mesa",
        "O livro está em cima da estante"
    ]
    labels = ['animal', 'animal', 'objeto', 'objeto']
    
    # Vetorizar textos
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(textos)
    
    # Treinar classificador
    modelo = MultinomialNB()
    modelo.fit(X, labels)
    
    # Testar com novas sentenças
    novas_sentencas = [
        "Um felino está sobre o carpete",
        "O celular está carregando na tomada"
    ]
    
    # Fazer predições
    X_novo = vectorizer.transform(novas_sentencas)
    predicoes = modelo.predict(X_novo)
    probabilidades = modelo.predict_proba(X_novo)
    
    print("\nPredições do classificador:")
    for sentenca, pred, prob in zip(novas_sentencas, predicoes, probabilidades):
        print(f"\nSentença: '{sentenca}'")
        print(f"Classe predita: {pred}")
        print(f"Probabilidades: {dict(zip(modelo.classes_, prob))}")

def main():
    print("Demonstração da diferença entre Sentence Encoder e Classificador")
    print("=" * 60)
    
    # Exemplo com Sentence Encoder
    exemplo_sentence_encoder()
    
    # Exemplo com Classificador
    exemplo_classificador()
    
    print("\n" + "=" * 60)
    print("\nPrincipais diferenças demonstradas:")
    print("1. Sentence Encoder:")
    print("   - Captura significado semântico")
    print("   - Entende que 'gato' e 'felino' são similares")
    print("   - Pode ser usado para várias tarefas")
    print("\n2. Classificador:")
    print("   - Foca em classificar em categorias")
    print("   - Não entende sinônimos (precisa de treino)")
    print("   - Só serve para a tarefa treinada")

if __name__ == "__main__":
    main() 