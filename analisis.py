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

        # Guardar el DataFrame completo con nombre claro
        key = f"{model}_{prompt}{'_json' if json_flag else ''}"
        dfs_modelos[key] = df.copy()

        # Calcular promedios de columnas A–E (opcional)
        cols = [c for c in ["A", "B", "C", "D", "E"] if c in df.columns]
        if len(cols) == 5:
            prom = df[cols].mean(axis=0)
            if model not in resultados:
                resultados[model] = {}
            resultados[model][f"{prompt}{'_json' if json_flag else ''}"] = prom

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

# Heatmap de la matriz de resultados de la categoria A
def plot_heatmap_matriz_resultados():
    """Genera un heatmap de la matriz de resultados."""
    plt.figure(figsize=(10, 6))
    pivot_table = matriz_resultados.reset_index().pivot(index='Modelo', columns='Prompt', values='A')
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Heatmap de Resultados Promedio (Columna A)")
    plt.ylabel("Modelo")
    plt.xlabel("Prompt")
    fig = plt.gcf()
    return fig


# Promedio general por modelo
promedio_modelos = matriz_resultados.groupby("Modelo")[["A", "B", "C", "D", "E"]].mean()

if df_expertos is not None:
    cols_expertos = [c for c in ["A", "B", "C", "D", "E"] if c in df_expertos.columns]
    promedio_expertos = df_expertos[cols_expertos].mean().to_frame().T
    promedio_expertos.index = ["Expertos"]
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
    plt.title("Promedio General por Modelo y Expertos (A–E)")
    plt.ylabel("Modelo / Expertos")
    plt.xlabel("Categorías")
    plt.tight_layout()
    return plt.gcf()

# Heatmap por prompt comparando con expertos de cada LLM

def plot_heatmap_prompt1_vs_expertos():
    """
    Genera un heatmap comparando el promedio de cada categoría (A–E)
    del Prompt 1 de cada modelo con el promedio de los expertos.
    """
    # === 1. Filtrar solo prompt 1 ===
    prompt1 = matriz_resultados.xs("p1", level="Prompt", drop_level=False)
    prompt1 = prompt1[["A", "B", "C", "D", "E"]]  # aseguramos columnas
    
    # === 2. Calcular promedio de expertos ===
    if df_expertos is not None:
        cols_expertos = [c for c in ["A", "B", "C", "D", "E"] if c in df_expertos.columns]
        prom_expertos = df_expertos[cols_expertos].mean().to_frame().T
        prom_expertos.index = ["Expertos"]
    else:
        print("⚠️ No se encontró hoja 'expertos'.")
        return None

    # === 3. Combinar ===
    comparacion_p1 = pd.concat([prompt1.droplevel("Prompt"), prom_expertos], axis=0)

    # === 4. Graficar ===
    plt.figure(figsize=(8, 5))
    sns.heatmap(comparacion_p1, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Comparación Prompt 1 vs Expertos (Categorías A–E)")
    plt.ylabel("Modelo / Expertos")
    plt.xlabel("Categorías")
    plt.tight_layout()

    return plt.gcf()

def plot_heatmap_prompt2_vs_expertos():
    """
    Genera un heatmap comparando el promedio de cada categoría (A–E)
    del Prompt 2 de cada modelo con el promedio de los expertos.
    """
    # === 1. Filtrar solo prompt 1 ===
    prompt2 = matriz_resultados.xs("p2", level="Prompt", drop_level=False)
    prompt2 = prompt2[["A", "B", "C", "D", "E"]]  # aseguramos columnas
    
    # === 2. Calcular promedio de expertos ===
    if df_expertos is not None:
        cols_expertos = [c for c in ["A", "B", "C", "D", "E"] if c in df_expertos.columns]
        prom_expertos = df_expertos[cols_expertos].mean().to_frame().T
        prom_expertos.index = ["Expertos"]
    else:
        print("⚠️ No se encontró hoja 'expertos'.")
        return None

    # === 3. Combinar ===
    comparacion_p2 = pd.concat([prompt2.droplevel("Prompt"), prom_expertos], axis=0)

    # === 4. Graficar ===
    plt.figure(figsize=(8, 5))
    sns.heatmap(comparacion_p2, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Comparación Prompt 2 vs Expertos (Categorías A–E)")
    plt.ylabel("Modelo / Expertos")
    plt.xlabel("Categorías")
    plt.tight_layout()

    return plt.gcf()

def plot_heatmap_prompt3_vs_expertos():
    """
    Genera un heatmap comparando el promedio de cada categoría (A–E)
    del Prompt 3 de cada modelo con el promedio de los expertos.
    """
    # === 1. Filtrar solo prompt 1 ===
    prompt3 = matriz_resultados.xs("p3", level="Prompt", drop_level=False)
    prompt3 = prompt3[["A", "B", "C", "D", "E"]]  # aseguramos columnas
    
    # === 2. Calcular promedio de expertos ===
    if df_expertos is not None:
        cols_expertos = [c for c in ["A", "B", "C", "D", "E"] if c in df_expertos.columns]
        prom_expertos = df_expertos[cols_expertos].mean().to_frame().T
        prom_expertos.index = ["Expertos"]
    else:
        print("⚠️ No se encontró hoja 'expertos'.")
        return None

    # === 3. Combinar ===
    comparacion_p3 = pd.concat([prompt3.droplevel("Prompt"), prom_expertos], axis=0)

    # === 4. Graficar ===
    plt.figure(figsize=(8, 5))
    sns.heatmap(comparacion_p3, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Comparación Prompt 1 vs Expertos (Categorías A–E)")
    plt.ylabel("Modelo / Expertos")
    plt.xlabel("Categorías")
    plt.tight_layout()

    return plt.gcf()

# Boxplot de todas las categorías para todos los modelos
def plot_boxplot_llm_por_prompt():
    """
    Genera un boxplot mostrando la distribución de las puntuaciones (A–E)
    para cada modelo y prompt.
    """
    # Convertir matriz_resultados a formato largo (long format)
    df_long = matriz_resultados.reset_index().melt(
        id_vars=["Modelo", "Prompt"],
        value_vars=["A", "B", "C", "D", "E"],
        var_name="Categoria",
        value_name="Puntuacion"
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_long, x="Modelo", y="Puntuacion", hue="Prompt")
    plt.title("Distribución de Puntuaciones por Modelo y Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Puntuación")
    plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return plt.gcf()

def plot_boxplot_llm_por_prompt_facets():
    """
    Muestra un boxplot separado por modelo (uno por panel) para cada prompt.
    """
    df_long = matriz_resultados.reset_index().melt(
        id_vars=["Modelo", "Prompt"],
        value_vars=["A", "B", "C", "D", "E"],
        var_name="Categoria",
        value_name="Puntuacion"
    )

    g = sns.catplot(
        data=df_long,
        x="Prompt",
        y="Puntuacion",
        col="Modelo",
        kind="box",
        col_wrap=3,
        sharey=True,
        height=4,
        aspect=1
    )
    for ax in g.axes.flat:
        ax.set_ylim(0, 5)
    g.fig.suptitle("Distribución de Puntuaciones por Prompt para Cada Modelo", y=1.05)
    return g.fig

# corregir scatterplot de promedio por ID
def plot_scatter_promedio_por_id(data):
    """
    Genera un scatterplot del promedio de puntuaciones (A–E) por ID
    del primer DataFrame en la lista 'data'.

    Parameters
    ----------
    data : list of pandas.DataFrame
        Lista de DataFrames (por ejemplo, generada a partir de dfs_modelos).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figura generada o None si no hay datos válidos.
    """
    if not data:
        print("⚠️ La lista 'data' está vacía.")
        return None

    df = data[0]
    if df.empty:
        print("⚠️ El primer DataFrame está vacío.")
        return None

    # Verificamos que exista la columna ID
    if "ID" not in df.columns:
        print("⚠️ No se encontró la columna 'ID' en el DataFrame.")
        print("Columnas disponibles:", list(df.columns))
        return None

    # Seleccionamos las columnas de puntuaciones
    cols = [c for c in ["A", "B", "C", "D", "E"] if c in df.columns]
    if not cols:
        print("⚠️ No se encontraron columnas A–E en el DataFrame.")
        return None

    # Calculamos el promedio por ID
    df_prom = df[["ID"] + cols].groupby("ID")[cols].mean()
    df_prom["Promedio"] = df_prom.mean(axis=1)

    # Graficar
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df_prom.index, y=df_prom["Promedio"])
    plt.title("Promedio de Puntuaciones (A–E) por ID")
    plt.xlabel("ID")
    plt.ylabel("Promedio")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt.gcf()



plot_functions = [
    plot_heatmap_matriz_resultados,
    plot_boxplot_llm_por_prompt_facets,
    lambda: plot_scatter_promedio_por_id(data)]

handle_plots(plot_functions, show=True, save=False, prefix="analisis_resultados")