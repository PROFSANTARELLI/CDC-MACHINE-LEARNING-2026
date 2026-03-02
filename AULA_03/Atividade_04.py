import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Dados de telemetria
data = {
    'intervalo_medio_ms': [150, 20, 180, 15, 200, 25, 165, 10],
    'erros_gramaticais': [5, 0, 8, 0, 4, 0, 6, 0],
    'origem': [0, 1, 0, 1, 0, 1, 0, 1] # 0: Humano, 1: Bot
}
df = pd.DataFrame(data)

# 2. Separação de dados
X = df[['intervalo_medio_ms', 'erros_gramaticais']]
y = df['origem']

# 3. Treinamento da Árvore
detector_bot = DecisionTreeClassifier(criterion='entropy') # 'entropy' mede a pureza da separação
detector_bot.fit(X, y)

# 4. Teste de Inferência
# Cenário A: Digitação ultra rápida e sem erros (Bot?)
print("Teste A (Rápido/Sem erro):", detector_bot.predict([[12, 0]])) 

# Cenário B: Digitação pausada e com erros (Humano?)
print("Teste B (Lento/Com erros):", detector_bot.predict([[210, 5]]))
