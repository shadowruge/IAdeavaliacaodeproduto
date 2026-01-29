import joblib
import nltk
import re
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------

nltk.download("stopwords", quiet=True)

MODEL_FILE = "modelo_pln.pkl"

# -------------------------------------------------------------------
# PRÉ-PROCESSAMENTO
# -------------------------------------------------------------------

def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúàâêôãõç\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# -------------------------------------------------------------------
# DATASET (ALINHADO E BALANCEADO)
# -------------------------------------------------------------------

positivos = [
    "excelente produto",
    "ótimo produto",
    "produto muito bom",
    "qualidade excelente",
    "funciona perfeitamente",
    "super recomendo",
    "gostei muito",
    "chegou rápido e bem embalado",
    "vale muito a pena",
    "produto sensacional",
    "ótima experiência",
    "atendeu minhas expectativas",
    "exelente produto",     # erro proposital
    "otimo produto",        # erro proposital
    "bom demais",
    "produto maravilhoso",
]

negativos = [
    "péssimo produto",
    "não gostei",
    "não funciona",
    "produto ruim",
    "qualidade péssima",
    "muito caro",
    "não vale a pena",
    "chegou quebrado",
    "demora na entrega",
    "suporte não responde",
    "experiência horrível",
    "dinheiro jogado fora",
    "produto defeituoso",
    "odiei",
    "horrível",
    "não recomendo",
]

textos = positivos + negativos
labels = ["positivo"] * len(positivos) + ["negativo"] * len(negativos)

# Normaliza os textos
textos = [normalizar_texto(t) for t in textos]

# -------------------------------------------------------------------
# PIPELINE INTELIGENTE
# -------------------------------------------------------------------

modelo = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char_wb",   # tolera erros de digitação
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
        ),
    ),
    (
        "clf",
        MultinomialNB(alpha=0.3)
    ),
])

# -------------------------------------------------------------------
# TREINAMENTO
# -------------------------------------------------------------------

modelo.fit(textos, labels)

# -------------------------------------------------------------------
# SALVAR MODELO
# -------------------------------------------------------------------

joblib.dump(modelo, MODEL_FILE)
print("✅ Modelo de PLN treinado e salvo com sucesso!")
