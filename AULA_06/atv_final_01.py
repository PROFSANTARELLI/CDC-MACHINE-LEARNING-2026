import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Configurações Iniciais
nltk.download('punkt')
nltk.download('stopwords')
stops = set(stopwords.words('portuguese'))

def pipeline_limpeza(texto):
    texto = texto.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(texto)
    return " ".join([w for w in tokens if w not in stops])

# --- DATASET DE TREINO PARA OS EXERCÍCIOS ---
faq_suporte = [
    "Como posso redefinir minha senha de acesso?",
    "Onde encontro o boleto para pagamento da fatura?",
    "Meu pedido está atrasado, quero saber o status da entrega.",
    "Não consigo fazer login no aplicativo, aparece erro 404.",
    "Quero cancelar minha assinatura e pedir estorno.",
    "Como atualizar meu endereço de entrega no cadastro?"
]

# ---------------------------------------------------------
# EXERCÍCIO 1: O "PESO" DA DIMENSIONALIDADE
# Objetivo: Entender como o vocabulário dita o tamanho da matriz.
# ---------------------------------------------------------
def exercicio_01():
    print("\n>>> EXERCÍCIO 1: ANÁLISE DE DIMENSIONALIDADE")
    
    # Tarefa: Adicione 2 frases novas e bem diferentes ao FAQ e veja o vocabulário crescer.
    faq_expandido = faq_suporte + [
        "quais as formas de pagamento aceitas",
        "como falar com um atendente humano"
    ]
    
    cv = CountVectorizer()
    matriz = cv.fit_transform(faq_expandido)
    
    print(f"Total de Documentos (Linhas): {matriz.shape[0]}")
    print(f"Total de Palavras Únicas (Colunas/Features): {matriz.shape[1]}")
    print(f"Vocabulário Gerado: {list(cv.get_feature_names_out()[:10])}... (e mais)")
    
    # PERGUNTA PARA O ALUNO: Se tivéssemos 1 milhão de frases, o que aconteceria com a memória do PC?

# ---------------------------------------------------------
# EXERCÍCIO 2: A SENSIBILIDADE DO TF-IDF
# Objetivo: Provar que palavras comuns perdem valor estatístico.
# ---------------------------------------------------------
def exercicio_02():
    print("\n>>> EXERCÍCIO 2: O FILTRO TF-IDF NA PRÁTICA")
    
    # Vamos injetar a palavra "ajuda" em TODAS as frases.
    faq_comum = [frase + " ajuda" for frase in faq_suporte]
    
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(faq_comum)
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    
    # Vamos verificar o peso da palavra 'ajuda' (que está em todas) 
    # versus a palavra 'senha' (que é rara).
    palavra_comum = 'ajuda'
    palavra_rara = 'senha'
    
    print(f"Peso TF-IDF da palavra '{palavra_comum}': {df_tfidf[palavra_comum].mean():.4f}")
    print(f"Peso TF-IDF da palavra '{palavra_rara}':  {df_tfidf[palavra_rara].max():.4f}")
    
    # CONCLUSÃO: O TF-IDF penaliza 'ajuda' porque ela não ajuda a distinguir uma frase da outra.

# ---------------------------------------------------------
# EXERCÍCIO 3: BUSCADOR DE FAQ (SIMILARIDADE DE COSSENO)
# Objetivo: Criar a lógica de um motor de busca semântico simples.
# ---------------------------------------------------------
def exercicio_03():
    print("\n>>> EXERCÍCIO 3: MOTOR DE BUSCA POR SIMILARIDADE")
    
    # 1. Vetorizamos o nosso FAQ original
    tfidf = TfidfVectorizer()
    X_faq = tfidf.fit_transform(faq_suporte)
    
    # 2. Pergunta do usuário (O que o bot acabou de receber)
    query_usuario = "esqueci minha senha e não consigo entrar"
    print(f"Pergunta do Usuário: '{query_usuario}'")
    
    # 3. Vetorizamos a pergunta usando o MESMO vocabulário do FAQ
    X_query = tfidf.transform([query_usuario])
    
    # 4. Cálculo da Similaridade de Cosseno (compara a query com todas as frases do FAQ)
    # Retorna uma lista de scores de 0 (nada a ver) a 1 (idêntico)
    similaridades = cosine_similarity(X_query, X_faq).flatten()
    
    # 5. Encontrar a melhor resposta
    indice_melhor = np.argmax(similaridades)
    score_maximo = similaridades[indice_melhor]
    
    print("-" * 30)
    print(f"Melhor Resposta Encontrada: {faq_suporte[indice_melhor]}")
    print(f"Confiança do Bot: {score_maximo * 100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    exercicio_01()
    exercicio_02()
    exercicio_03()
