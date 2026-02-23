import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar Dados
df = pd.read_csv('chatbot_data.csv')

# 2. Separar Features (X) e Alvo (y)
X = df.drop('label', axis=1) # Características da mensagem
y = df['label']              # O que queremos prever

# 3. Divisão Treino/Teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar o Modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. Avaliar
predicoes = model.predict(X_test)
print(f"Acurácia do Modelo: {accuracy_score(y_test, predicoes) * 100:.2f}%")

# 6. Teste Real
nova_msg = [[120, 1, 4]] # [comprimento, exclamacao, palavras_negativas]
resultado = model.predict(nova_msg)
print("Resultado da Previsão (0: Dúvida | 1: Reclamação):", resultado[0])
