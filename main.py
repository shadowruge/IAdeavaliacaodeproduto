from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import logging

# -------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# -------------------------------------------------------------------

APP_NAME = "Analisador de Sentimentos"
MODEL_FILE = "modelo_pln.pkl"
BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------
# LOGGING (útil para debug e produção)
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
)

# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção: definir domínios
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# MODELO
# -------------------------------------------------------------------

model_path = BASE_DIR / MODEL_FILE

if not model_path.exists():
    logger.critical("Arquivo do modelo não encontrado: %s", model_path)
    raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

try:
    modelo = joblib.load(model_path)
    logger.info("Modelo carregado com sucesso")
except Exception as e:
    logger.critical("Erro ao carregar modelo: %s", e)
    raise

# -------------------------------------------------------------------
# SCHEMAS
# -------------------------------------------------------------------

class Entrada(BaseModel):
    texto: str = Field(..., min_length=3, description="Texto para análise de sentimento")

class Saida(BaseModel):
    sentimento: str
    confianca: str

# -------------------------------------------------------------------
# ROTAS
# -------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
def serve_home():
    index_file = BASE_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html não encontrado")

    return index_file


@app.post("/analisar", response_model=Saida)
def analisar(dados: Entrada):
    try:
        pred = modelo.predict([dados.texto])[0]

        # Confiança (se existir)
        if hasattr(modelo, "predict_proba"):
            probs = modelo.predict_proba([dados.texto])
            confianca = max(probs[0])
        else:
            confianca = 1.0

        # Normalização do sentimento
        sentimento = str(pred).lower()
        if sentimento in {"1", "positivo", "positive", "pos"}:
            sentimento = "positivo"
        else:
            sentimento = "negativo"

        return {
            "sentimento": sentimento,
            "confianca": f"{confianca:.2%}",
        }

    except Exception as e:
        logger.error("Erro ao analisar texto: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar o texto",
        )

# -------------------------------------------------------------------
# HEALTH CHECK (para deploy)
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}
