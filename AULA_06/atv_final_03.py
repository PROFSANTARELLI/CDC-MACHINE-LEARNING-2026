import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import string

# (pré-processamento simplificado)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download the missing 'punkt_tab' resource
stops = set(stopwords.words('portuguese'))

def limpar_simplificado(texto):
    texto = texto.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(texto, language='portuguese') # Explicitly specify Portuguese
    tokens_limpos = [w for w in tokens if w not in stops and w.isalnum()]
    return " ".join(tokens_limpos)

# 1. DADOS DE ENTRADA (Dataset Grande de Logs Não Rotulados)
# Imagine que estas são 15 mensagens novas que o bot recebeu
mensagens_logs = [
    "meu boleto nao chegou", "queria pagar com pix", "erro no login nao entra",
    "atraso no pedido entrega demora", "cartao recusado pagamento falhou",
    "login bloqueado senha invalida", "frete muito caro para o meu cep",
    "quero cancelar a compra", "boleto vencido como pagar",
    "pix nao funciona erro no app", "pedido nao foi entregue prazo venceu",
    "cancelamento de pedido como fazer", "valor do frete cep 01001",
    "senha esquecida nao consigo login", "estorno do pagamento cancelado"
]

df = pd.DataFrame({'mensagem_bruta': mensagens_logs})
df['mensagem_limpa'] = df['mensagem_bruta'].apply(limpar_simplificado)

# 2. VETORIZAÇÃO
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['mensagem_limpa'])

# 3. EXERCÍCIO: CLUSTERIZAÇÃO (K-Means)
# Defina o número de clusters (grupos) que você deseja encontrar
# Testem com 3, 4 e 5 clusters
N_CLUSTERS = 4

print(f"--- Executando K-Means para encontrar {N_CLUSTERS} grupos ---")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X)

# Adiciona o rótulo do cluster de volta ao DataFrame
df['cluster'] = kmeans.labels_

# 4. ANÁLISE DOS RESULTADOS
# Vamos imprimir as mensagens de cada cluster para ver se o agrupamento faz sentido
for i in range(N_CLUSTERS):
    print(f"\n[CLUSTER {i}] - Tópico Sugerido:")
    messages_in_cluster = df[df['cluster'] == i]['mensagem_bruta'].tolist()

    # Imprime as primeiras 4 mensagens do cluster
    for msg in messages_in_cluster[:4]:
        print(f"  - {msg}")

# OBSERVAÇÃO TÉCNICA:
# Provavelmente o K-Means agrupará mensagens sobre:
# Cluster X: Pagamentos (boleto, pix, cartao)
# Cluster Y: Logística/Entrega (atraso, pedido, frete, cep)
# Cluster Z: Acesso (login, senha, bloqueado)
# Cluster W: Cancelamento (cancelar, cancelado, estorno)
