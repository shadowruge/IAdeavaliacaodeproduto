# 🤖 IA de Avaliação de Produto

Aplicação web de **Análise de Sentimentos em Português** para avaliar opiniões de clientes sobre produtos, utilizando **Inteligência Artificial**, **FastAPI** e **Machine Learning**.

O sistema permite que o usuário digite uma avaliação textual e receba, em tempo real:
- Classificação do sentimento (positivo ou negativo)
- Grau de confiança da IA
- Histórico das análises
- Gráfico estatístico de sentimentos

---

## 📌 Demonstração (local)

- Front-end: `index.html`
- Backend: FastAPI (`/analisar`)
- Comunicação via JSON (fetch API)

---

## 🧠 Tecnologias Utilizadas

### Backend
- Python 3
- FastAPI
- Scikit-learn
- NLTK
- Joblib
- Uvicorn

### Machine Learning
- TF-IDF (caracteres + n-gramas)
- Multinomial Naive Bayes
- Pré-processamento customizado para português
- Tolerância a erros de digitação

### Frontend
- HTML5
- CSS3 (UI moderna, dark mode)
- JavaScript (fetch + localStorage)
- Chart.js (gráfico de sentimentos)

---

2️⃣ Crie e ative um ambiente virtual (opcional, recomendado)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

3️⃣ Instale as dependências
pip install -r backend/requirements.txt

4️⃣ Treine o modelo de IA
python backend/treinar.py
Isso irá gerar o arquivo:

modelo_pln.pkl

5️⃣ Inicie a API FastAPI
uvicorn backend.main:app --reload


A API ficará disponível em:

http://127.0.0.1:8000

6️⃣ Abra o Front-end

Abra o arquivo:

frontend/index.html


no navegador.

🔗 Endpoint da API
POST /analisar

Request

{
  "texto": "excelente produto"
}


Response

{
  "sentimento": "positivo",
  "confianca": "92.31%"
}

📊 Funcionalidades Implementadas

✅ Análise de sentimentos em português

✅ Tolerância a erros ortográficos

✅ Histórico persistente (localStorage)

✅ Gráfico estatístico (Chart.js)

✅ UI moderna (dark mode)

✅ API REST pronta para deploy

🔮 Próximas Evoluções Planejadas

🔐 Login de usuários

📈 Linha do tempo de sentimentos

🤖 Classe “neutro”

🧠 Modelo BERTimbau

🐳 Dockerização

☁️ Deploy em nuvem (Render)

👨‍💻 Autor

Projeto desenvolvido por Izaias de Oliveira Elias
GitHub: https://github.com/shadowruge

📄 Licença

Este projeto é open-source e pode ser utilizado para fins educacionais e experimentais.




```

