import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Cargar todas las hojas
xls = pd.read_excel("Resultados LLM.xlsx", sheet_name=None)  # dict {hoja: DataFrame}

# 2) Quitar las hojas que no quieres
excluded = {"Transcripciones", "Expertos"}
sheet_names = [name for name in xls.keys() if name not in excluded]

# 3) Serie con promedios de expertos (asegúrate que la hoja Expertos existe)
if "Expertos" not in xls:
    raise ValueError("No se encontró la hoja 'Expertos' en el archivo.")
df_expertos = xls["Expertos"]
promedios_expertos = df_expertos[["A","B","C","D","E"]].mean(axis=0)  # Series con índice A..E

# Regex para extraer: modelo, prompt y optional _json
# Ejemplos que captura: "ChatGPT_p1", "Gemini_p2_json", "Llama_p10_json"
pattern = re.compile(r'^\s*(?P<model>.+?)_p(?P<prompt>\d+)(?:_(?P<json>json))?\s*$', flags=re.IGNORECASE)

def process_sheet(df):
    """Devuelve una Serie con índices A..E (prom_p1) o None si no hay suficientes datos."""
    cols_p1 = [c for c in ["A","B","C","D","E"] if c in df.columns]
    prom_p1 = None
    if len(cols_p1) == 5:
        prom_p1 = df[cols_p1].mean(axis=0)
    elif df.shape[1] >= 5:  # fallback por posición
        prom_p1 = df.iloc[:, 0:5].mean(axis=0)
        prom_p1.index = ["A","B","C","D","E"]
    return prom_p1

frames = []
skipped_sheets = []
parsed_info = []  # para mostrar qué se interpretó

for sheet in sheet_names:
    m = pattern.match(sheet)
    if not m:
        # si el nombre no sigue el patrón, lo saltamos (o podrías intentar un parse más laxo)
        skipped_sheets.append(sheet)
        continue

    model = m.group("model").strip()
    prompt_num = m.group("prompt")
    json_flag = bool(m.group("json"))

    # procesar hoja
    df_sheet = xls[sheet]
    prom1 = process_sheet(df_sheet)
    if prom1 is None:
        skipped_sheets.append(sheet)
        continue

    # crear DF temporal con la info extraída
    df_temp = pd.DataFrame({
        "Categoria": prom1.index,
        "Promedio": prom1.values,
        "Expertos": promedios_expertos.reindex(prom1.index).values,
        "LLM": model,
        "Prompt": prompt_num,
        "JSON": json_flag
    })
    frames.append(df_temp)
    parsed_info.append((sheet, model, prompt_num, json_flag))

# Concatenar resultados (si hay)
if frames:
    df_final = pd.concat(frames, ignore_index=True).dropna(subset=["Promedio","Expertos"])
else:
    df_final = pd.DataFrame(columns=["Categoria","Promedio","Expertos","LLM","Prompt","JSON"])

# Informes de verificación
print("Hojas procesadas (con parse aplicado):")
for info in parsed_info:
    print("  - hoja:", info[0], "=> model:", info[1], "prompt:", info[2], "json:", info[3])

if skipped_sheets:
    print("\nHojas saltadas (no coincidieron con el patrón o faltan columnas):")
    for s in skipped_sheets:
        print("  -", s)

print("\nData final (primeras filas):")
print(df_final.head())

print("\nEstadísticas generales:")
print("Media Promedio:", df_final["Promedio"].mean())
print("Media Expertos:", df_final["Expertos"].mean())

# --- Preparación / limpieza adicional para graficar ---
# Asegurarnos de que Categoria tenga el orden correcto
ordered_cats = ["A","B","C","D","E"]
df_final["Categoria"] = pd.Categorical(df_final["Categoria"], categories=ordered_cats, ordered=True)

# Convertir Prompt a entero (si es posible) para facilitar orden y tamaño
try:
    df_final["Prompt"] = df_final["Prompt"].astype(int)
except:
    # si hay valores no convertibles, dejar como string
    df_final["Prompt"] = df_final["Prompt"].astype(str)

# Normalizar la columna JSON a etiquetas legibles
df_final["JSON"] = df_final["JSON"].map({True: "json", False: "no_json"}).fillna("no_json")

# Mostrar conteos rápidos para verificar
print("Conteo por Prompt y JSON:\n", df_final.groupby(["Prompt","JSON"]).size())

# ----------------- GRAFICO 1: catplot por Prompt (columnas) y JSON (filas) -----------------
# Usar col_wrap por si hay muchos prompts, evita que se estire demasiado horizontalmente
n_prompts = df_final["Prompt"].nunique()
col_wrap = 4 if n_prompts > 4 else n_prompts

g = sns.catplot(
    data=df_final,
    x="Categoria", y="Promedio",
    hue="LLM",
    col="Prompt", row="JSON",
    kind="point",
    dodge=True,
    height=4, aspect=1,
    order=ordered_cats
)
g.fig.subplots_adjust(top=0.92)
g.fig.suptitle("Promedio por Categoría — dividido por Prompt y Formato (json/no_json)")
plt.show()


# ----------------- GRAFICO 2: scatter horizontal Promedio vs Categoria -----------------
try:
    df_final["Prompt"] = df_final["Prompt"].astype(int)
except:
    # si no puede convertirlo, crea una versión numérica temporal basada en categoría
    df_final["Prompt_num"] = pd.Categorical(df_final["Prompt"]).codes + 1
    df_final["Prompt"] = df_final["Prompt_num"]

# Normalizar JSON a etiquetas legibles (si no lo hiciste ya)
df_final["JSON"] = df_final["JSON"].map({True: "json", False: "no_json"}).fillna("no_json")

# Scatterplot usando size="Prompt" y rango de tamaños
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_final,
    x="Promedio", y="Categoria",
    hue="LLM", style="JSON",
    size="Prompt", sizes=(50, 300),  # rango de tamaños (min, max)
    alpha=0.9,
    palette="tab10"
)
plt.title("Promedio vs Categoria — color: LLM | estilo: JSON | tamaño: Prompt")
plt.xlim(left=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()
