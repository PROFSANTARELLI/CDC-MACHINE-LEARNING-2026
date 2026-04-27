import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def inicializar_projeto():
    print(" Iniciando a preparação do sistema...")

    # --- 1. GERADOR DE DATASET (1000 LINHAS) ---
    np.random.seed(42)
    n = 1000
    data = {
        'nome': [f"Paciente {i}" for i in range(n)],
        'idade': np.random.randint(18, 90, n),
        'glicose': np.random.randint(70, 250, n),
        'pressao_arterial': np.random.randint(80, 180, n),
        'imc': np.random.uniform(18, 45, n),
        'colesterol': np.random.randint(150, 350, n)
    }
    df = pd.DataFrame(data)

    # Lógica de Risco (Regra de Negócio para o Treino)
    df['risco'] = ((df['glicose'] > 126) | (df['imc'] > 32) | ((df['idade'] > 70) & (df['pressao_arterial'] > 140))).astype(int)

    df.to_csv('dados_clinicos.csv', index=False)
    print(" Dataset 'dados_clinicos.csv' criado.")

    # --- 2. TREINAMENTO DO MODELO IA ---
    X = df[['idade', 'glicose', 'pressao_arterial', 'imc', 'colesterol']]
    y = df['risco']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'modelo_saude.pkl')
    print(" IA Treinada: 'modelo_saude.pkl' gerado.")

    # --- 3. BANCO DE DADOS SQLITE ---
    conn = sqlite3.connect('clinica.db')
    cursor = conn.cursor()

    # Limpeza e Criação de Tabelas
    cursor.execute('DROP TABLE IF EXISTS usuarios')
    cursor.execute('CREATE TABLE usuarios (user TEXT PRIMARY KEY, password TEXT)')
    cursor.execute('INSERT INTO usuarios VALUES ("admin", "1234"), ("medico", "senha123")')

    cursor.execute('DROP TABLE IF EXISTS pacientes')
    cursor.execute('''CREATE TABLE pacientes
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT, idade INTEGER,
                       glicose REAL, pressao_arterial REAL, imc REAL, colesterol REAL, risco INTEGER)''')

    cursor.execute('DROP TABLE IF EXISTS tipos_exames')
    cursor.execute('''CREATE TABLE tipos_exames
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, nome_exame TEXT, unidade TEXT, v_alerta REAL)''')

    # Inserção de dados padrão
    exames_padrao = [("Glicose", "mg/dL", 126.0), ("Colesterol", "mg/dL", 240.0), ("IMC", "kg/m²", 30.0)]
    cursor.executemany('INSERT INTO tipos_exames (nome_exame, unidade, v_alerta) VALUES (?,?,?)', exames_padrao)

    # Migração do CSV para o SQL
    df.to_sql('pacientes', conn, if_exists='append', index=False)

    conn.commit()
    conn.close()
    print(" Banco de Dados 'clinica.db' configurado e populado.")

if __name__ == "__main__":
    inicializar_projeto()
