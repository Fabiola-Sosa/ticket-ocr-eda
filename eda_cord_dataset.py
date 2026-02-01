"""
EDA del Proyecto: Extracción y Análisis de Tickets de Venta (CORD-1K)
Autor: Fabiola Sosa
Descripción:
    - Extrae el dataset desde un ZIP
    - Parsea los archivos JSON del dataset CORD-1K
    - Construye un DataFrame estructurado
    - Realiza un EDA completo para análisis exploratorio
"""

import os
import json
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. CONFIGURACIONES Y RUTAS
# ---------------------------------------------------------

ZIP_PATH = "cord_dataset.zip"          # Cambiar si usas otro nombre del ZIP
EXTRACT_DIR = "cord_dataset"           # Carpeta donde se extraerá
FINAL_DATASET_DIR = os.path.join(EXTRACT_DIR, "CORD")

# ---------------------------------------------------------
# 2. EXTRACCIÓN DEL ZIP
# ---------------------------------------------------------

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"📦 Extrayendo {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        print(f"✔️ Dataset extraído en: {extract_to}")
    else:
        print(f"✔️ Carpeta ya existe: {extract_to}")

extract_zip(ZIP_PATH, EXTRACT_DIR)

# ---------------------------------------------------------
# 3. PARSEADOR DE JSON CORD
# ---------------------------------------------------------

def parse_cord1k_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = data["meta"]["image_id"]

    items = []
    quantities = []
    prices = []

    total = None
    cash = None
    change = None

    for line in data["valid_line"]:
        category = line["category"]
        words = [w["text"] for w in line["words"]]
        text_joined = " ".join(words)

        if category == "menu.nm":
            items.append(text_joined)

        elif category == "menu.cnt":
            quantities.append(text_joined)

        elif category == "menu.price":
            prices.append(text_joined)

        elif category == "total.total_price":
            total = words[-1]

        elif category == "total.cashprice":
            cash = words[-1]

        elif category == "total.changeprice":
            change = words[-1]

    return {
        "image_id": image_id,
        "items": items,
        "quantities": quantities,
        "prices": prices,
        "total": total,
        "cash": cash,
        "change": change,
    }

# ---------------------------------------------------------
# 4. Cargar todo el dataset
# ---------------------------------------------------------

def load_full_cord_dataset(base_dir):
    rows = []
    for split in ["train", "dev", "test"]:
        json_dir = os.path.join(base_dir, split, "json")

        for file in os.listdir(json_dir):
            if file.endswith(".json"):
                row = parse_cord1k_json(os.path.join(json_dir, file))
                row["split"] = split
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


df = load_full_cord_dataset(FINAL_DATASET_DIR)
print("✔️ DataFrame construido con éxito")
print(df.head())


# ---------------------------------------------------------
# 5. CONVERSIÓN DE CAMPOS NUMÉRICOS
# ---------------------------------------------------------

def clean_numeric(x):
    if isinstance(x, list):
        return [pd.to_numeric(v, errors="coerce") for v in x]
    return pd.to_numeric(x, errors="coerce")

df["prices_num"] = df["prices"].apply(clean_numeric)
df["quantities_num"] = df["quantities"].apply(clean_numeric)
df["total_num"] = df["total"].apply(clean_numeric)
df["cash_num"] = df["cash"].apply(clean_numeric)
df["change_num"] = df["change"].apply(clean_numeric)


# ---------------------------------------------------------
# 6. EDA COMPLETO
# ---------------------------------------------------------

print("\n=====  RESUMEN GENERAL DEL DATASET  =====")
print(df.info())
print(df.describe(include="all"))

print("\n===== NULOS =====")
print(df.isnull().sum())

print("\n===== Distribución por split =====")
print(df["split"].value_counts())

# ---------------------------------------------------------
# 7. VISUALIZACIONES
# ---------------------------------------------------------

sns.set(style="whitegrid")

# Histograma: Total de pago
plt.figure(figsize=(8,5))
df["total_num"].dropna().hist(bins=30)
plt.title("Distribución del total de los tickets")
plt.xlabel("Total")
plt.ylabel("Frecuencia")
plt.show()

# Cantidad de items por ticket
df["num_items"] = df["items"].apply(len)

plt.figure(figsize=(8,5))
sns.countplot(x=df["num_items"])
plt.title("Cantidad de items por ticket")
plt.xlabel("Número de items")
plt.ylabel("Frecuencia")
plt.show()

# Heatmap de nulos
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Mapa de valores nulos")
plt.show()

# ---------------------------------------------------------
# 8. EXPORT FINAL DEL DATAFRAME LIMPIO
# ---------------------------------------------------------

OUTPUT_PATH = "cord_dataset_clean.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✔️ Archivo exportado: {OUTPUT_PATH}")

