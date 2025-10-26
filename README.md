# 🧠 Previsão de Seguro de Vida

Este projeto utiliza aprendizado de máquina (Machine Learning) para prever o custo de um seguro de vida, com base em informações pessoais e de saúde.
O projeto inclui um dashboard interativo feito com Streamlit e um modelo treinado com Random Forest.

🚀 Funcionalidades

Limpeza e preparação dos dados automatizada por meio de um pipeline de dados

Treinamento e otimização de um modelo Random Forest Regressor

Análise de importância das variáveis (feature importance)

Dashboard interativo para realizar previsões a partir dos dados inseridos pelo usuário

🧰 Tecnologias Utilizadas

Python (pandas, numpy, scikit-learn, matplotlib, seaborn)

Streamlit (para criação do dashboard)

Pickle (para salvar e carregar o modelo treinado)

📊 Dashboard

O dashboard permite que o usuário insira informações como idade, IMC, região e número de filhos para obter uma previsão do custo do seguro.

Para rodar o dashboard, execute o comando abaixo no terminal:

streamlit run app.py

📂 Estrutura do Projeto
previsao_seguro_de_vida/
│
├── data/
│   └── insurance.csv
│
├── model/
│   ├── best_forest_model.pkl
│   └── full_pipeline.pkl
│
├── app.py
├── notebook_analise.ipynb
└── README.md

🧪 Treinamento do Modelo

O arquivo notebook_analise.ipynb contém:

Exploração e visualização dos dados

Análise de correlação

Preparação e transformação das variáveis

Treinamento e avaliação do modelo

Salvamento do modelo final com Pickle

🔮 Importância das Variáveis

O modelo Random Forest foi utilizado para identificar as variáveis que mais impactam no valor do seguro.
As principais foram:

Status de fumante (smoker)

Idade (age)

IMC (bmi)