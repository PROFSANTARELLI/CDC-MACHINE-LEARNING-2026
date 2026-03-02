import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# DICA 1: Use pd.read_csv para trazer o arquivo 'dados_intencoes.csv'
df = pd.read_csv('dados_intencoes.csv')

# DICA 2: X deve conter todas as colunas exceto 'label'. y é apenas 'label'.
X = df.drop('label', axis=1)
y = df['label']

# --- INÍCIO DO TRABALHO DO ALUNO ---
# PASSO 1: Divida os dados em treino e teste (use test_size=0.2)
# PASSO 2: Instancie o DecisionTreeClassifier e treine com .fit()
# PASSO 3: Gere predições e use classification_report(y_test, predicoes) para ver o resultado por classe.
