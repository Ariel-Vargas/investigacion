import re
import pandas as pd  # type: ignore
import numpy as np
import seaborn as sns # type: ignore
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
print(df_final)

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
# n_prompts = df_final["Prompt"].nunique()
# col_wrap = 4 if n_prompts > 4 else n_prompts

# g = sns.catplot(
#     data=df_final,
#     x="Categoria", y="Promedio",
#     hue="LLM",
#     col="Prompt", row="JSON",
#     kind="point",
#     dodge=True,
#     height=4, aspect=1,
#     order=ordered_cats
# )
# g.fig.subplots_adjust(top=0.92)
# g.fig.suptitle("Promedio por Categoría — dividido por Prompt y Formato (json/no_json)")
# plt.show()


# # ----------------- GRAFICO 2: scatter horizontal Promedio vs Categoria -----------------
# try:
#     df_final["Prompt"] = df_final["Prompt"].astype(int)
# except:
#     # si no puede convertirlo, crea una versión numérica temporal basada en categoría
#     df_final["Prompt_num"] = pd.Categorical(df_final["Prompt"]).codes + 1
#     df_final["Prompt"] = df_final["Prompt_num"]

# # Normalizar JSON a etiquetas legibles (si no lo hiciste ya)
# df_final["JSON"] = df_final["JSON"].map({True: "json", False: "no_json"}).fillna("no_json")

# # Scatterplot usando size="Prompt" y rango de tamaños
# plt.figure(figsize=(10, 6))
# sns.scatterplot(
#     data=df_final,
#     x="Promedio", y="Categoria",
#     hue="LLM", style="JSON",
#     size="Prompt", sizes=(50, 300),  # rango de tamaños (min, max)
#     alpha=0.9,
#     palette="tab10"
# )
# plt.title("Promedio vs Categoria — color: LLM | estilo: JSON | tamaño: Prompt")
# plt.xlim(left=0)
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# plt.tight_layout()
# plt.show()


# # --- Preparar datos en formato largo para violinplot ---
# df_violin = df_final.melt(
#     id_vars=["Categoria", "LLM", "Prompt", "JSON"],
#     value_vars=["Promedio", "Expertos"],
#     var_name="Fuente",
#     value_name="Valor"
# )

# # --- Graficar violinplot ---
# plt.figure(figsize=(10, 6))
# sns.violinplot(
#     data=df_violin,
#     x="Categoria", y="Valor",
#     hue="Fuente",            # comparamos LLM vs Expertos
#     split=True,              # pone ambos en el mismo violín
#     palette="Set2"
# )

# plt.title("Distribución de valores por Categoría — Comparación LLM vs Expertos")
# plt.ylim(0, 5)  # opcional, si tus valores están en escala 0-5
# plt.legend(title="Fuente")
# plt.tight_layout()
# plt.show()



######## Empezar aquí #########
# ----------------- GRAFICO 1: scatter con flechas por Modelo -----------------
# plt.figure(figsize=(8,6))

# # Asegurarnos de que los promedios sean numéricos
# df_final["Expertos"] = pd.to_numeric(df_final["Expertos"], errors="coerce")
# df_final["Promedio"] = pd.to_numeric(df_final["Promedio"], errors="coerce")

# # Agrupar por modelo y formato
# df_avg = df_final.groupby(["LLM","JSON"])[["Expertos","Promedio"]].mean().reset_index()

# # Colores por modelo (puedes personalizar)
# colores = {
#     "ChatGPT": "blue",
#     "Gemini": "orange",
#     "Claude": "green",
#     "Deepseek": "purple",
#     "Grok": "brown",
#     "Mistral": "red"
# }

# # Dibujar cada modelo
# for modelo in df_avg["LLM"].unique():
#     subset = df_avg[df_avg["LLM"] == modelo]

#     if set(subset["JSON"]) >= {"no_json","json"}:
#         # coordenadas
#         x0, y0 = subset[subset["JSON"]=="no_json"][["Expertos","Promedio"]].values[0]
#         x1, y1 = subset[subset["JSON"]=="json"][["Expertos","Promedio"]].values[0]

#         # Flecha (limpia, sin relleno gris)
#         plt.annotate("",
#             xy=(x1, y1), xytext=(x0, y0),
#             arrowprops=dict(arrowstyle="->", color=colores.get(modelo,"gray"), lw=1.5)
#         )

#         # Puntos
#         plt.scatter(x0, y0, marker="o", color=colores.get(modelo,"gray"), label="no_json" if modelo==df_avg["LLM"].unique()[0] else "")
#         plt.scatter(x1, y1, marker="*", color=colores.get(modelo,"gray"), s=120, label="json" if modelo==df_avg["LLM"].unique()[0] else "")

#         # Texto del modelo junto al punto final
#         plt.text(x1+0.03, y1, modelo, fontsize=9)

# plt.xlabel("Expertos (%)")
# plt.ylabel("Promedio LLM (%)")
# plt.title("Comparación Expertos vs LLM (flecha: no_json → json)")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()


# # ----------------- GRAFICO 2: scatter con flechas por Categoria -----------------

# plt.figure(figsize=(10,7))

# # Asegurarnos que sean numéricos
# df_final["Expertos"] = pd.to_numeric(df_final["Expertos"], errors="coerce")
# df_final["Promedio"] = pd.to_numeric(df_final["Promedio"], errors="coerce")

# # Agrupar por modelo, categoría y formato
# df_avg = df_final.groupby(["LLM","Categoria","JSON"])[["Expertos","Promedio"]].mean().reset_index()

# # Colores por modelo
# colores = {
#     "ChatGPT": "blue",
#     "Gemini": "orange",
#     "Claude": "green",
#     "Deepseek": "purple",
#     "Grok": "brown",
#     "Mistral": "red"
# }

# # Dibujar cada modelo y categoría
# for modelo in df_avg["LLM"].unique():
#     for cat in df_avg["Categoria"].unique():
#         subset = df_avg[(df_avg["LLM"]==modelo) & (df_avg["Categoria"]==cat)]

#         if set(subset["JSON"]) >= {"no_json","json"}:
#             # coordenadas
#             x0, y0 = subset[subset["JSON"]=="no_json"][["Expertos","Promedio"]].values[0]
#             x1, y1 = subset[subset["JSON"]=="json"][["Expertos","Promedio"]].values[0]

#             # Flecha
#             plt.annotate("",
#                 xy=(x1, y1), xytext=(x0, y0),
#                 arrowprops=dict(arrowstyle="->", color=colores.get(modelo,"gray"), lw=1)
#             )

#             # Puntos
#             plt.scatter(x0, y0, marker="o", color=colores.get(modelo,"gray"), alpha=0.7)
#             plt.scatter(x1, y1, marker="*", color=colores.get(modelo,"gray"), s=100, alpha=0.9)

#             # Texto con modelo + categoría al final de la flecha
#             plt.text(x1+0.03, y1, f"{modelo}-{cat}", fontsize=8)

# plt.xlabel("Expertos (%)")
# plt.ylabel("Promedio LLM (%)")
# plt.title("Comparación Expertos vs LLM por Categoría (flecha: no_json → json)")
# plt.grid(alpha=0.3)
# plt.show()



# ----------------- GRAFICO 3: boxplot Promedio vs Categoria -----------------
sns.boxplot(
    data=df_final,
    x="Categoria", y="Promedio",
    hue="LLM"
)
plt.title("Distribución de Promedios por Categoría y LLM")
plt.show()


# ----------------- GRAFICO 4: boxplot Promedio vs Categoria (solo ChatGPT) -----------------
df_llm = df_final[df_final["LLM"] == "ChatGPT"]
sns.boxplot(
    data=df_llm,
    x="Categoria", y="Promedio",
    hue="Prompt",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
)
for i in range(len(df_final["Categoria"].unique()) - 1):
    plt.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.7)

plt.title("Distribución de Promedios por Categoría — ChatGPT (3 prompts)")
plt.show()


#----------------- GRAFICO 5: boxplot Promedio vs Categoria (solo Gemini) -----------------
df_llm = df_final[df_final["LLM"] == "Gemini"]
sns.boxplot(
    data=df_llm,
    x="Categoria", y="Promedio",
    hue="Prompt",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
)
for i in range(len(df_final["Categoria"].unique()) - 1):
    plt.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.7)

plt.title("Distribución de Promedios por Categoría — Gemini (3 prompts)")
plt.show()

#----------------- GRAFICO 6: boxplot Promedio vs Categoria (solo Deepseek) -----------------
df_llm = df_final[df_final["LLM"] == "Deepseek"]
sns.boxplot(
    data=df_llm,
    x="Categoria", y="Promedio",
    hue="Prompt",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
)
for i in range(len(df_final["Categoria"].unique()) - 1):
    plt.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.7)

plt.title("Distribución de Promedios por Categoría — Deepseek (3 prompts)")
plt.show()


#----------------- GRAFICO 6: boxplot Promedio vs Categoria (solo Meta) -----------------
df_llm = df_final[df_final["LLM"] == "Meta"]
sns.boxplot(
    data=df_llm,
    x="Categoria", y="Promedio",
    hue="Prompt",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
)
for i in range(len(df_final["Categoria"].unique()) - 1):
    plt.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.7)

plt.title("Distribución de Promedios por Categoría — Meta (3 prompts)")
plt.show()


#----------------- GRAFICO 7: boxplot Promedio vs Categoria (solo Claude) -----------------
df_llm = df_final[df_final["LLM"] == "Claude"]
sns.boxplot(
    data=df_llm,
    x="Categoria", y="Promedio",
    hue="Prompt",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
)
for i in range(len(df_final["Categoria"].unique()) - 1):
    plt.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.7)

plt.title("Distribución de Promedios por Categoría — Claude (3 prompts)")
plt.show()


#----------------- GRAFICO 8: boxplot Promedio vs Categoria (solo Grok) -----------------
df_llm = df_final[df_final["LLM"] == "Grok"]
sns.boxplot(
    data=df_llm,
    x="Categoria", y="Promedio",
    hue="Prompt",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
)
for i in range(len(df_final["Categoria"].unique()) - 1):
    plt.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.7)

plt.title("Distribución de Promedios por Categoría — Grok (3 prompts)")
plt.show()

# ----------------- GRAFICO 9: boxplot Promedio vs Categoria (facetas por LLM) -----------------
g = sns.catplot(
    data=df_final,
    x="Categoria", y="Promedio",
    hue="Prompt",
    kind="box",
    col="LLM",
    col_wrap=3,
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
    height=4, aspect=1
)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Distribución de Promedios por Categoría y Prompt (por LLM)")
plt.show()


