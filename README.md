# 💡 Previsão de Custo de Seguro de Saúde

Este projeto tem como objetivo **prever o custo anual e mensal de um seguro de saúde** com base em informações fornecidas pelo usuário, como idade, IMC, número de filhos, sexo, tabagismo e região.  
Foi desenvolvido utilizando **Python, Scikit-Learn, Pandas e Streamlit**.

---

## 🧠 Objetivo

Criar um modelo de aprendizado de máquina capaz de estimar o custo do seguro de saúde de uma pessoa, considerando fatores de risco e características individuais.  
Além disso, o projeto inclui um **aplicativo interativo em Streamlit** que permite ao usuário inserir seus dados e visualizar a previsão em tempo real.

---

## 🧩 Tecnologias Utilizadas

- **Python 3.10+**
- **Pandas** — tratamento e análise de dados  
- **NumPy** — operações numéricas e estatísticas  
- **Matplotlib** — visualização dos dados  
- **Scikit-Learn** — criação e treino do modelo  
- **Streamlit** — criação da interface interativa  
- **Pickle** — salvar e carregar o modelo treinado  

---

## ⚙️ Como Executar o Projeto

1. **Clone o repositório**
   ```bash
   git clone https://github.com/ViniciusFreitas06/previsao_seguro_de_vida.git
   cd previsao_seguro_de_vida
2. **Crie e ative um ambiente virtual (opcional, mas recomendado)**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows
3. **Instale as dependências**
    ```bash
    pip install -r requirements.txt
4. **Execute o app**
    ```bash
    streamlit run app.py
---

## 📊 Modelos Testados

Durante o desenvolvimento, dois modelos foram comparados:

| Modelo                 | RMSE (Erro Médio Quadrático) |
|-------------------------|------------------------------|
| Regressão Linear        | ~6000                        |
| Random Forest Regressor | ~4400                        |

O modelo **Random Forest** apresentou melhor desempenho e foi escolhido como modelo final.

---

## 🖥️ Funcionalidades do App

- Inserção manual dos dados pelo usuário (idade, IMC, filhos, etc.);
- Exibição da previsão **anual e mensal** do seguro;
- Visualização das **importâncias das variáveis** no modelo;
- Gráficos interativos mostrando a relação entre **variáveis numéricas e o custo do seguro**.

---

## 🧾 Exemplo de Uso

**Exemplo de previsão gerada:**

- **Idade:** 30 anos  
- **IMC:** 25.0  
- **Filhos:** 1  
- **Sexo:** Masculino  
- **Fumante:** Não  
- **Região:** Southeast  

**Resultado:**
- 💰 **Custo anual estimado:** US$ 4.820,50  
- 💵 **Custo mensal estimado:** US$ 401,71
