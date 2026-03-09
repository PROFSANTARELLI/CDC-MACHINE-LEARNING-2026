import numpy as np
import matplotlib.pyplot as plt

# 1. DEFINIÇÃO DOS DADOS (O que aconteceu na realidade vs O que o modelo previu)
# y_real representa os valores verdadeiros (Ground Truth)
y_real = np.array([10.0, 20.0, 30.0]) 

# y_hat representa o valor PREVISTO pelo modelo (ŷ)
y_hat = np.array([12.0, 18.0, 40.0]) 

# 2. CÁLCULO DOS RESÍDUOS (A diferença individual)
# Subtraímos o valor real do previsto para encontrar o erro de cada ponto
residuos = y_real - y_hat

# 3. ALGORITMO DO MAE (Mean Absolute Error)
# Passo A: Transformamos todos os erros em valores positivos (absolutos)
erros_absolutos = np.abs(residuos)
# Passo B: Calculamos a média desses valores positivos
mae = np.mean(erros_absolutos)

# 4. ALGORITMO DO RMSE (Root Mean Squared Error)
# Passo A: Elevamos os erros ao quadrado (isso elimina o sinal e penaliza erros grandes)
erros_quadraticos = residuos ** 2
# Passo B: Calculamos a média dos erros quadráticos (MSE)
mse = np.mean(erros_quadraticos)
# Passo C: Extraímos a raiz quadrada para voltar à unidade de medida original
rmse = np.sqrt(mse)

# 5. EXIBIÇÃO DOS RESULTADOS NO TERMINAL
print("--- Demonstração do Algoritmo ---")
print(f"Valores Reais (y):      {y_real}")
print(f"Valores Previstos (ŷ):  {y_hat}")
print("-" * 33)
print(f"MAE calculado:  {mae:.2f}")
print(f"RMSE calculado: {rmse:.2f}")

# 6. VISUALIZAÇÃO GRÁFICA (A distância como erro)
plt.figure(figsize=(10, 6))

# Plota os valores reais como pontos azuis
plt.scatter(range(len(y_real)), y_real, color='blue', label='Valor Real (y)', s=100, zorder=3)
# Plota as previsões como "X" vermelhos
plt.scatter(range(len(y_hat)), y_hat, color='red', marker='x', label='Valor Previsto (ŷ)', s=100, zorder=3)

# Desenha linhas verticais para representar o erro (distância)
for i in range(len(y_real)):
    plt.vlines(i, y_real[i], y_hat[i], colors='gray', linestyles='dashed', alpha=0.5)
    plt.text(i + 0.05, (y_real[i] + y_hat[i])/2, f' Erro: {abs(residuos[i])}', color='darkred')

# Configurações do Gráfico
plt.title('Visualização de Erros: Real vs Previsto')
plt.xlabel('Amostras / Clientes')
plt.ylabel('Valores')
plt.xticks(range(len(y_real)), ['Cliente A', 'Cliente B', 'Cliente C'])
plt.legend()
plt.grid(True, axis='y', alpha=0.3)

# Exibe o gráfico
plt.show()

# OBSERVAÇÃO TÉCNICA:
# Note no gráfico que a distância do 'Cliente C' é a maior.
# No RMSE, essa distância de 10 unidades vira 100 no cálculo, elevando a média final.
