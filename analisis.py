import re
import pandas as pd  # type: ignore
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

def handle_plots(plot_funcs, show=True, save=False, prefix="plot"):
    """
    Ejecuta una lista de funciones que devuelven figuras y las muestra/guarda.

    Parameters
    ----------
    plot_funcs : list of callables
        Cada función debe devolver un objeto matplotlib.figure.Figure
    show : bool
        Si True, se muestran las figuras
    save : bool
        Si True, se guardan como PNG
    prefix : str
        Prefijo para los archivos PNG
    """
    figures = []

    for func in plot_funcs:
        fig = func()
        if fig is not None:
            figures.append(fig)

    if save:
        for i, fig in enumerate(figures, 1):
            filename = f"{prefix}_{i}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✅ Guardado: {filename}")

    if show:
        for fig in figures:
            plt.figure(fig.number)
        plt.show()


xls = pd.read_excel("Resultados LLM.xlsx", sheet_name=None)
pattern = re.compile(r'^\s*(?P<model>.+?)_p(?P<prompt>\d+)(?:_(?P<json>json))?\s*$', flags=re.IGNORECASE)

def process_sheet(df):
    cols_p1 = [c for c in ["A","B","C","D","E"] if c in df.columns]
    prom_p1 = None
    if len(cols_p1) == 5:
        prom_p1 = df[cols_p1].mean(axis=0)
    elif df.shape[1] >= 5:
        prom_p1 = df.iloc[:, 0:5].mean(axis=0)
        prom_p1.index = ["A","B","C","D","E"]
    return prom_p1


# Diccionarios de salida
dfs_modelos = {}     # Cada hoja (chatgpt_p1, claude_p1, etc.)
resultados = {}     # Aquí se guardan los promedios
df_expertos = None  # DataFrame de la hoja "expertos"


# Recorremos todas las hojas
for sheet_name, df in xls.items():
    sheet_name_clean = sheet_name.strip().lower()

    # Hoja de expertos
    if sheet_name_clean == "expertos":
        df_expertos = df.copy()
        continue

    # Match con el patrón de modelos
    match = pattern.match(sheet_name_clean)
    if match:
        model = match.group("model").strip().lower()
        prompt = f"p{match.group('prompt')}"
        json_flag = match.group("json") is not None

        # ⚙️ Aquí reemplazamos "_json" por "_batch" en el nombre clave
        key = f"{model}_{prompt}{'_batch' if json_flag else ''}"
        dfs_modelos[key] = df.copy()

        # Calcular promedios de columnas A–E (opcional)
        cols = [c for c in ["A", "B", "C", "D", "E"] if c in df.columns]
        if len(cols) == 5:
            prom = df[cols].mean(axis=0)
            if model not in resultados:
                resultados[model] = {}
            resultados[model][f"{prompt}{'_batch' if json_flag else ''}"] = prom

# Convertimos los resultados en una matriz (DataFrame combinando todo) y se muestra el promedio por modelo y prompt
matriz_resultados = pd.concat(
    {
        model: pd.DataFrame.from_dict(prompts, orient="index")
        for model, prompts in resultados.items()
    },
    names=["Modelo", "Prompt"]
)

data = [] # Lista de todos los DataFrames
for name in dfs_modelos.keys():
    data.append(dfs_modelos.get(name, pd.DataFrame()))

df_concatenado = pd.concat(data, ignore_index=True) # DataFrame combinado de todos los modelos

column_mapping = {
    "A": "A - Estructura del discurso",
    "B": "B - Cohesión y Coherencia",
    "C": "C - Calidad del vocabulario y uso del Lenguaje",
    "D": "D - Argumentación y evidencia",
    "E": "E - Estrategias discursivas y manejo del tema"
}

df_concatenado.rename(columns=column_mapping, inplace=True)

# También aplicar el cambio a todos los DataFrames individuales
for key in dfs_modelos.keys():
    dfs_modelos[key].rename(columns=column_mapping, inplace=True)

# Y si existe df_expertos, también
if df_expertos is not None:
    df_expertos.rename(columns=column_mapping, inplace=True)

# Si también quieres que la matriz_resultados y comparaciones usen los nuevos nombres:
matriz_resultados.rename(columns=column_mapping, inplace=True)

print("✅ DataFrame concatenado de todos los modelos:")
print(df_concatenado.head())
print("\nDimensiones del DataFrame concatenado:", df_concatenado.shape)

# === Mostrar resultados ===
# print("✅ DataFrames cargados:")
# for name in dfs_modelos.keys():
#     print("  -", name)
#     print("\n=== Ejemplo: DataFrame de", name, "===")
#     print(dfs_modelos.get(name, pd.DataFrame()).head())

print("\n=== DataFrame de expertos ===")
print(df_expertos.head() if df_expertos is not None else "No se encontró hoja 'expertos'.")

print("\n=== Matriz de resultados (promedios por modelo/prompt) ===")
print(matriz_resultados.head())


#### Análisis y visualización ####

# Nombres actualizados para las columnas
cols_descriptivas = [
    "A - Estructura del discurso",
    "B - Cohesión y Coherencia",
    "C - Calidad del vocabulario y uso del Lenguaje",
    "D - Argumentación y evidencia",
    "E - Estrategias discursivas y manejo del tema"
]
# Promedio general por modelo
promedio_modelos = matriz_resultados.groupby("Modelo")[cols_descriptivas].mean()

if df_expertos is not None:
    cols_expertos = [c for c in cols_descriptivas if c in df_expertos.columns]
    promedio_expertos = df_expertos[cols_expertos].mean().to_frame().T
    promedio_expertos.index = ["Evaluadores \nhumanos"]
else:
    promedio_expertos = pd.DataFrame()

# Combinar en un solo DataFrame
comparacion = pd.concat([promedio_modelos, promedio_expertos])
print("\n=== Promedio general por modelo y expertos ===")
print(comparacion)


def plot_heatmap_promedios_generales():
    """Genera un heatmap del promedio general de cada modelo y expertos."""
    plt.figure(figsize=(8, 5))
    sns.heatmap(comparacion, annot=True, fmt=".2f", cmap="YlOrBr")
    plt.title("Promedio General por Evaluador")
    plt.tight_layout()
    return plt.gcf()

# Heatmap por prompt comparando con expertos de cada LLM

def plot_heatmap_prompt_vs_expertos(prompt_num):
    """
    Genera un heatmap comparando el promedio de cada categoría (A–E)
    del Prompt indicado con el promedio de los expertos.
    """
    prompt_label = f"p{prompt_num}"
    prompt_df = matriz_resultados.xs(prompt_label, level="Prompt", drop_level=False)
    prompt_df = prompt_df[cols_descriptivas]  # aseguramos columnas correctas
    
    # Calcular promedio de expertos
    if df_expertos is not None:
        cols_expertos = [c for c in cols_descriptivas if c in df_expertos.columns]
        prom_expertos = df_expertos[cols_expertos].mean().to_frame().T
        prom_expertos.index = ["Evaluadores humanos"]
    else:
        print("⚠️ No se encontró hoja 'expertos'.")
        return None

    comparacion_prompt = pd.concat([prompt_df.droplevel("Prompt"), prom_expertos], axis=0)

    # Crear etiquetas multilínea para el eje X
    categorias_multilinea = [
        "A - Estructura\ndel discurso",
        "B - Cohesión\ny Coherencia",
        "C - Calidad del\nvocabulario y uso del Lenguaje",
        "D - Argumentación\ny evidencia",
        "E - Estrategias\ndiscursivas y manejo del tema"
    ]

    comparacion_prompt.columns = categorias_multilinea

    plt.figure(figsize=(12, 6))
    sns.heatmap(comparacion_prompt, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Comparación Prompt {prompt_num} vs Expertos (Categorías A–E)")
    plt.ylabel("Modelo / Expertos")
    plt.xlabel("Categorías")
    plt.xticks(rotation=0, ha="center")
    plt.yticks(ha="right")
    plt.tight_layout()

    return plt.gcf()

# Funciones individuales para cada prompt
def plot_heatmap_prompt1_vs_expertos():
    return plot_heatmap_prompt_vs_expertos(1)

def plot_heatmap_prompt2_vs_expertos():
    return plot_heatmap_prompt_vs_expertos(2)

def plot_heatmap_prompt3_vs_expertos():
    return plot_heatmap_prompt_vs_expertos(3)



# Boxplot de todas las categorías para todos los modelos
def plot_boxplot_llm_por_prompt():
    """
    Genera un boxplot mostrando la distribución de las puntuaciones (A–E)
    para cada modelo y prompt.
    """
    # Convertir matriz_resultados a formato largo (long format)
    df_long = matriz_resultados.reset_index().melt(
        id_vars=["Modelo", "Prompt"],
        value_vars=cols_descriptivas,
        var_name="Categoria",
        value_name="Puntuacion"
    )

    colores = ["#155DFC", 
               "#FF5733", 
               "#33FF57",
               "#2B7FFF",
               "#FF8904",
               "#7BF1A8"
              ]

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_long, x="Modelo", y="Puntuacion", hue="Prompt", width=0.6, palette=colores)
    modelos = df_long["Modelo"].unique()
    for i in range(len(modelos) - 1):
        ax.axvline(x=i + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.title("Distribución de Puntuaciones por Modelo y Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Puntuación")
    plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return plt.gcf()


plot_functions = [
    plot_heatmap_prompt1_vs_expertos,
    plot_heatmap_prompt2_vs_expertos,
    plot_heatmap_prompt3_vs_expertos,
    plot_boxplot_llm_por_prompt
    ]

handle_plots(plot_functions, show=True, save=False, prefix="analisis_resultados")