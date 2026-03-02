import pandas as pd
import numpy as np

def gerar_dados_intencoes(n=300):
    data = {
        'qtd_palavras': np.random.randint(2, 20, n),
        'contem_valor_monetario': np.random.choice([0, 1], n),
        'uso_termos_tecnicos': np.random.randint(0, 5, n),
        'intensidade_pontuacao': np.random.randint(0, 10, n),
        'label': [] # 0: Suporte, 1: Comercial, 2: Crítica
    }
    for i in range(n):
        if data['intensidade_pontuacao'][i] > 7:
            data['label'].append(2) # Crítica (Urgente)
        elif data['contem_valor_monetario'][i] == 1:
            data['label'].append(1) # Comercial (Vendas)
        else:
            data['label'].append(0) # Suporte (Dúvidas)
    return pd.DataFrame(data)

df = gerar_dados_intencoes()
df.to_csv('dados_intencoes.csv', index=False)
print(" Arquivo 'dados_intencoes.csv' gerado!")
