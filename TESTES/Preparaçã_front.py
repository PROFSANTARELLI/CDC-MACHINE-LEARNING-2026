%%writefile app.py
import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os

# --- CONFIGURAÇÕES DE PÁGINA ---
st.set_page_config(page_title="Sistema de Análise Clínica", page_icon="🏥", layout="wide")

# --- FUNÇÕES AUXILIARES DE BANCO DE DADOS ---
def run_query(query, params=()):
    with sqlite3.connect('clinica.db') as conn:
        return pd.read_sql_query(query, conn, params=params)

def execute_db(query, params=()):
    with sqlite3.connect('clinica.db') as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()

# --- CARREGAMENTO DO MODELO IA ---
@st.cache_resource
def load_model():
    if os.path.exists('modelo_saude.pkl'):
        return joblib.load('modelo_saude.pkl')
    return None

# --- TELAS DO SISTEMA ---

def tela_login():
    st.title("🏥 Acesso ao Sistema Clínico")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login"):
            st.subheader("Autenticação de Utilizador")
            u = st.text_input("Utilizador")
            p = st.text_input("Palavra-passe", type="password")
            if st.form_submit_button("Entrar", use_container_width=True):
                res = run_query("SELECT * FROM usuarios WHERE user=? AND password=?", (u, p))
                if not res.empty:
                    st.session_state.auth = True
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("Credenciais inválidas")

def tela_cadastrar_paciente():
    st.header("👤 Cadastro de Novos Pacientes")
    with st.form("form_pac"):
        nome = st.text_input("Nome do Paciente")
        idade = st.number_input("Idade", 0, 120)
        if st.form_submit_button("Guardar Paciente"):
            if nome:
                execute_db("INSERT INTO pacientes (nome, idade) VALUES (?,?)", (nome, idade))
                st.success(f"Paciente {nome} registado!")
            else:
                st.error("Nome obrigatório")

def tela_gestao_exames():
    st.header("⚙️ Configuração de Exames e Riscos")
    with st.expander("➕ Adicionar Novo Tipo de Exame", expanded=True):
        with st.form("form_exame"):
            nome = st.text_input("Nome do Exame")
            uni = st.text_input("Unidade (ex: mg/dL)")
            v_min = st.number_input("Mínimo Referência", 0.0)
            v_max = st.number_input("Máximo Referência", 0.0)
            v_alerta = st.number_input("Valor de Alerta de Risco", 0.0)
            if st.form_submit_button("Gravar Configuração"):
                execute_db("INSERT INTO tipos_exames (nome_exame, unidade, v_min, v_max, v_alerta) VALUES (?,?,?,?,?)", 
                           (nome, uni, v_min, v_max, v_alerta))
                st.success("Configuração salva!")
                st.rerun()
    
    st.subheader("📋 Tabela de Referências Atuais")
    df_ex = run_query("SELECT * FROM tipos_exames")
    st.dataframe(df_ex, use_container_width=True)

def tela_insercao_resultados():
    st.header("🧪 Lançamento de Resultados e Análise de IA")
    
    # Seleção do Paciente
    pacientes_df = run_query("SELECT id, nome FROM pacientes")
    if pacientes_df.empty:
        st.warning("Cadastre um paciente primeiro.")
        return

    pac_escolhido = st.selectbox("Selecione o Paciente", pacientes_df['nome'])
    pac_id = pacientes_df[pacientes_df['nome'] == pac_escolhido]['id'].values[0]
    
    with st.form("analise_ia"):
        st.write("Insira os valores dos exames colhidos:")
        c1, col2 = st.columns(2)
        with c1:
            idade = st.number_input("Idade Atual", 18, 100)
            glicose = st.number_input("Glicose (mg/dL)", 0.0)
            pressao = st.number_input("Pressão Arterial", 0.0)
        with col2:
            imc = st.number_input("IMC", 0.0)
            colesterol = st.number_input("Colesterol Total", 0.0)
            
        if st.form_submit_button("🚀 EXECUTAR ANÁLISE PREDITIVA"):
            model = load_model()
            if model:
                # Preparação para a IA
                dados = pd.DataFrame([[idade, glicose, pressao, imc, colesterol]], 
                                     columns=['idade', 'glicose', 'pressao_arterial', 'imc', 'colesterol'])
                
                pred = model.predict(dados)[0]
                prob = model.predict_proba(dados).max()
                resultado_texto = "ALTO RISCO" if pred == 1 else "NORMAL / BAIXO RISCO"
                
                # Exibição de Resultado
                st.divider()
                st.subheader(f"📊 Laudo Gerado: {resultado_texto}")
                if pred == 1:
                    st.error(f"Atenção: Probabilidade de Risco de {prob*100:.2f}%")
                else:
                    st.success(f"Estabilidade detectada (Confiança: {prob*100:.2f}%)")
                
                # Guardar resultado no banco
                execute_db('''INSERT INTO resultados_exames 
                              (paciente_id, glicose, pressao, imc, colesterol, risco_detectado) 
                              VALUES (?,?,?,?,?,?)''', 
                           (int(pac_id), glicose, pressao, imc, colesterol, resultado_texto))
                st.info("Resultado salvo no histórico do paciente.")

def tela_dashboard():
    st.title("🏠 Menu Principal - Dashboard")
    c1, c2, c3 = st.columns(3)
    
    try:
        pacs = run_query("SELECT count(*) as total FROM pacientes")['total'][0]
        exs = run_query("SELECT count(*) as total FROM resultados_exames")['total'][0]
        alertas = run_query("SELECT count(*) as total FROM resultados_exames WHERE risco_detectado='ALTO RISCO'")['total'][0]
        
        c1.metric("Pacientes", pacs)
        c2.metric("Análises Feitas", exs)
        c3.metric("Casos Críticos", alertas, delta_color="inverse")
        
        st.subheader("🕒 Últimas Análises Realizadas")
        df_hist = run_query('''SELECT p.nome, r.glicose, r.risco_detectado, r.data_exame 
                               FROM resultados_exames r 
                               JOIN pacientes p ON p.id = r.paciente_id 
                               ORDER BY r.data_exame DESC LIMIT 5''')
        st.table(df_hist)
    except:
        st.info("Inicie o banco de dados e faça as primeiras análises para ver as métricas.")

# --- LÓGICA DE NAVEGAÇÃO ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    tela_login()
else:
    st.sidebar.title("MENU CLÍNICO")
    opcao = st.sidebar.radio("Navegação", 
                            ["Dashboard", "Cadastrar Paciente", "Gestão de Exames", "Lançar Resultados / IA"])
    
    if st.sidebar.button("Sair"):
        st.session_state.auth = False
        st.rerun()

    if opcao == "Dashboard": tela_dashboard()
    elif opcao == "Cadastrar Paciente": tela_cadastrar_paciente()
    elif opcao == "Gestão de Exames": tela_gestao_exames()
    elif opcao == "Lançar Resultados / IA": tela_insercao_resultados()
