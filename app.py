import streamlit as st
import pandas as pd
import joblib

# Layout da página
st.title("🩺 Previsão de Doenças Cardíacas")
st.write("Insira os dados clínicos do paciente para realizar a triagem.")

# Carregar os arquivos salvos do modelo
try:
    modelo = joblib.load('modelo_final.joblib')
    scaler = joblib.load('scaler.joblib')
    colunas = joblib.load('colunas_X.joblib')
except Exception as e:
    st.error("Erro ao carregar o modelo. Verifique os arquivos .joblib.")
    st.stop()

st.header("Dados do Paciente")
entradas = {}

# Criar os campos de entrada automaticamente
cols_st = st.columns(3)
for i, col in enumerate(colunas):
    with cols_st[i % 3]:
        # Para facilitar a apresentação, usamos input numérico
        entradas[col] = st.number_input(f"{col}", value=0.0)

# Botão de Executar
if st.button("Executar Predição"):
    # Transformar a entrada do usuário em DataFrame
    df_entrada = pd.DataFrame([entradas])
    
    # Escalonar
    X_scaled = scaler.transform(df_entrada)
    
    # Prever
    predicao = modelo.predict(X_scaled)
    
    # Exibir Resultado
    st.subheader("Resultado da Predição:")
    if predicao[0] == 1:
        st.error("🚨 ALERTA: Probabilidade de doença cardíaca detectada.")
    else:
        st.success("✅ Paciente sem risco imediato detectado pelo modelo.") 