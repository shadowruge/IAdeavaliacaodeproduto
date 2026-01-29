import joblib

modelo = joblib.load("modelo_pln.pkl")

tests = [
    "exelente produto",
    "otimo produto",
    "produto horrivell",
    "nao gostei",
    "vale muito a pena",
]

for t in tests:
    print(t, "→", modelo.predict([t])[0])
