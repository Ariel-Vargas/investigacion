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
        key = f"{model}_{prompt}{'-Batch' if json_flag else ''}"
        dfs_modelos[key] = df.copy()

        # Calcular promedios de columnas A–E (opcional)
        cols = [c for c in ["A", "B", "C", "D", "E"] if c in df.columns]
        if len(cols) == 5:
            prom = df[cols].mean(axis=0)
            if model not in resultados:
                resultados[model] = {}
            resultados[model][f"{prompt}{'-Batch' if json_flag else ''}"] = prom

# Convertimos los resultados en una matriz (DataFrame combinando todo) y se muestra el promedio por modelo y prompt
matriz_resultados = pd.concat(
    {
        model: pd.DataFrame.from_dict(prompts, orient="index")
        for model, prompts in resultados.items()
    },
    names=["Modelo", "Prompt"]
)

# === Renombrar modelos para visualización ===
nombre_modelos = {
    "chatgpt": "GPT-5",
    "gemini": "Gemini 2.5 flash",
    "deepseek": "Deepseek 3.2",
    "meta": "Llama 4",
    "claude": "Claude 3",
    "grok": "Grok 3"
}

# Aplicar los nombres nuevos a la matriz principal
matriz_resultados.rename(index=nombre_modelos, level="Modelo", inplace=True)

# También renombrar las claves en el diccionario 'resultados' si deseas mantener coherencia al imprimir
resultados = {
    nombre_modelos.get(model, model): prompts
    for model, prompts in resultados.items()
}


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

# Promedio general por modelo
promedio_modelos = matriz_resultados.groupby("Modelo")[["A", "B", "C", "D", "E"]].mean()

if df_expertos is not None:
    cols_expertos = [c for c in ["A", "B", "C", "D", "E"] if c in df_expertos.columns]
    promedio_expertos = df_expertos[cols_expertos].mean().to_frame().T
    promedio_expertos.index = ["Human"]
else:
    promedio_expertos = pd.DataFrame()

# Combinar en un solo DataFrame
comparacion = pd.concat([promedio_modelos, promedio_expertos])
print("\n=== Promedio general por modelo y expertos ===")
print(comparacion)

# === Diferencia promedio entre cada modelo y los expertos ===
if "Human" in comparacion.index:
    experto = comparacion.loc["Human"]
    
    # Calcular la diferencia por modelo (modelo - expertos)
    diferencias = comparacion.drop("Human").apply(lambda row: row - experto, axis=1)
    diferencias = diferencias.round(2)
    print("\n=== Diferencia promedio (modelo - expertos) por categoría ===")
    print(diferencias)
else:
    print("\n⚠️ No se encontró fila de expertos para calcular diferencias.")


def plot_heatmap_diferencias():
    """
    Genera un heatmap que muestra la diferencia promedio entre cada modelo y los expertos
    por categoría (A–E). Los valores positivos indican que hay una gran diferencia que
    los expertos, y los negativos indican lo contrario.
    """
    if "Human" not in comparacion.index:
        print("⚠️ No se encontró fila de expertos para calcular diferencias.")
        return None

    experto = comparacion.loc["Human"]
    diferencias = comparacion.drop("Human").apply(lambda row: row - experto, axis=1)
    diferencias = diferencias.round(2)

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 5))

    # Heatmap con mapa de color divergente (rojos negativos, azules positivos)
    sns.heatmap(
        diferencias,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",  # colores de rojo (negativo) a azul (positivo)
        center=0,
        ax=ax
    )

    ax.set_title("Difference between LLM qualifications based on expert rating")
    ax.set_xlabel("Category")
    ax.set_ylabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), ha="right")
    plt.tight_layout()

    # Ajustar espacio para la leyenda lateral
    fig.subplots_adjust(right=0.75)

    # Crear eje auxiliar para la descripción de categorías
    ax_legend = fig.add_axes([0.78, 0.2, 0.2, 0.6])
    ax_legend.axis("off")

    descripcion_categorias = {
        "A": "Speech Structure",
        "B": "Cohesion and Coherence",
        "C": "Vocabulary Quality \n and Language Use",
        "D": "Argumentation and Evidence",
        "E": "Discursive Strategies \n and Topic Management"
    }

    y_pos = 1.0
    y_step = 0.18
    ax_legend.text(0, y_pos + y_step * 0.5, "Category", fontsize=10, fontweight="bold")
    for i, (k, v) in enumerate(descripcion_categorias.items()):
        ax_legend.text(0, y_pos - i * y_step, f"{k} – {v}", fontsize=9, va="top")

    return fig


def plot_heatmap_promedios_generales():
    """Genera un heatmap del promedio general de cada modelo y expertos, con leyenda separada."""
    
    # Mantener solo letras como columnas
    categorias_letras = ["A", "B", "C", "D", "E"]
    df_heatmap = comparacion.copy()
    df_heatmap.columns = categorias_letras

    # Crear figura y eje principal
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Heatmap
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax)
    ax.set_title("Overall Average by Evaluator")
    ax.set_ylabel("Model / Human")
    ax.set_xlabel("Categorías")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), ha="right")
    
    # Ajuste de espacio a la derecha para la leyenda
    fig.subplots_adjust(right=0.75)  # dejar 25% para la leyenda

    # Crear eje auxiliar para la leyenda
    ax_legend = fig.add_axes([0.78, 0.2, 0.2, 0.6])  # [left, bottom, width, height]
    ax_legend.axis("off")  # eje invisible
    
    # Descripción de categorías
    descripcion_categorias = {
        "A": "Speech Structure",
        "B": "Cohesion and Coherence",
        "C": "Vocabulary Quality \n and Language Use",
        "D": "Argumentation and Evidence",
        "E": "Discursive Strategies \n and Topic Management"
    }

    # Mostrar texto de la leyenda
    y_pos = 1.0
    y_step = 0.18
    ax_legend.text(0, y_pos + y_step*0.5, "Category", fontsize=10, fontweight="bold")
    for i, (k, v) in enumerate(descripcion_categorias.items()):
        ax_legend.text(0, y_pos - i*y_step, f"{k} – {v}", fontsize=9, va="top")

    return fig

# Heatmap por prompt comparando con expertos de cada LLM

def plot_heatmap_prompt_vs_expertos(prompt_num):
    """
    Genera un heatmap comparando el promedio de cada categoría (A–E)
    del Prompt indicado con el promedio de los expertos.
    """
    prompt_label = f"p{prompt_num}"
    prompt_df = matriz_resultados.xs(prompt_label, level="Prompt", drop_level=False)
    prompt_df = prompt_df[["A", "B", "C", "D", "E"]]  # aseguramos columnas correctas
    
    # Calcular promedio de expertos
    if df_expertos is not None:
        cols_expertos = [c for c in ["A", "B", "C", "D", "E"] if c in df_expertos.columns]
        prom_expertos = df_expertos[cols_expertos].mean().to_frame().T
        prom_expertos.index = ["Human"]
    else:
        print("⚠️ No se encontró hoja 'expertos'.")
        return None

    comparacion_prompt = pd.concat([prompt_df.droplevel("Prompt"), prom_expertos], axis=0)

    # Mantener solo letras como columnas
    categorias_letras = ["A", "B", "C", "D", "E"]
    comparacion_prompt.columns = categorias_letras

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(12, 5))

    # Heatmap
    sns.heatmap(comparacion_prompt, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Analysis of prompt {prompt_num} ratings among human evaluator ratings")
    ax.set_ylabel("Model / Human")
    ax.set_xlabel("Category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), ha="right")

    # Ajuste de espacio a la derecha
    fig.subplots_adjust(right=0.75)  # dejar 25% para la leyenda

    # Crear un eje auxiliar para la leyenda
    ax_legend = fig.add_axes([0.78, 0.2, 0.2, 0.6])  # [left, bottom, width, height]
    ax_legend.axis("off")  # eje invisible

    descripcion_categorias = {
        "A": "Speech Structure",
        "B": "Cohesion and Coherence",
        "C": "Vocabulary Quality \n and Language Use",
        "D": "Argumentation and Evidence",
        "E": "Discursive Strategies \n and Topic Management"
    }

    # Mostrar texto de la leyenda
    y_pos = 1.0
    y_step = 0.18
    ax_legend.text(0, y_pos + y_step*0.5, "Category", fontsize=10, fontweight="bold")
    for i, (k, v) in enumerate(descripcion_categorias.items()):
        ax_legend.text(0, y_pos - i*y_step, f"{k} – {v}", fontsize=9, va="top")

    return fig

    # Funciones individuales para cada prompt
def plot_heatmap_prompt1_vs_expertos():
    return plot_heatmap_prompt_vs_expertos(1)

def plot_heatmap_prompt2_vs_expertos():
    return plot_heatmap_prompt_vs_expertos(2)

def plot_heatmap_prompt3_vs_expertos():
    return plot_heatmap_prompt_vs_expertos(3)



# Boxplot de todas las categorías para todos los modelos
def plot_boxplot_llm_por_prompt():

    # Convertir matriz_resultados a formato largo (long format)
    df_long = matriz_resultados.reset_index().melt(
        id_vars=["Modelo", "Prompt"],
        value_vars=["A", "B", "C", "D", "E"],
        var_name="Categoria",
        value_name="Puntuacion"
    )

    colores = ["#4C72B0", "#DD8452", "#55A868","#6A8CC7", "#E59C6D", "#7CC79A"]

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_long, x="Modelo", y="Puntuacion", hue="Prompt", width=0.6, palette=colores)
    modelos = df_long["Modelo"].unique()
    for i in range(len(modelos) - 1):
        ax.axvline(x=i + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.title("Distribution of scores for the Large Language Model")
    plt.xlabel("Large Language Model")
    plt.ylabel("Score")
    plt.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return plt.gcf()



# def plot_violin_chatgpt():
#     modelo_objetivo = "chatgpt"
    
#     # Filtrar solo el modelo ChatGPT
#     df_chatgpt = matriz_resultados.loc[modelo_objetivo]
#     df_chatgpt = df_chatgpt.reset_index()  # Recuperamos "Prompt" como columna

#     # Convertir a formato largo (long format)
#     df_long = df_chatgpt.melt(
#         id_vars=["Prompt"],
#         value_vars=["A", "B", "C", "D", "E"],
#         var_name="Categoria",
#         value_name="Puntuacion"
#     )

#     # Crear el gráfico
#     plt.figure(figsize=(10, 6))
#     sns.violinplot(
#         data=df_long,
#         x="Modelo",
#         y="Puntuacion",
#         hue="Categoria",
#         inner="box",   # Muestra la mediana y los cuartiles dentro del violín
#         split=True,
#         density_norm="width",
#     )

#     plt.title("Distribución de Puntuaciones de ChatGPT por Categoría")
#     plt.xlabel("Categorías")
#     plt.ylabel("Puntuación")
#     plt.xticks(rotation=15, ha="right")
#     plt.tight_layout()
#     return plt.gcf()


# def plot_cat_violin_todos_llm():
#     """
#     Genera un catplot (violinplot) mostrando la distribución de puntuaciones (A–E)
#     para cada modelo LLM en paneles separados.
#     """
#     # Convertir la matriz de resultados a formato largo (long format)
#     df_long = matriz_resultados.reset_index().melt(
#         id_vars=["Modelo", "Prompt"],
#         value_vars=["A", "B", "C", "D", "E"],
#         var_name="Categoria",
#         value_name="Puntuacion"
#     )

#     # Crear el gráfico tipo catplot (violinplot)
#     g = sns.catplot(
#         data=df_long,
#         x="Categoria",
#         y="Puntuacion",
#         kind="violin",
#         col="Modelo",            # Cada modelo en un panel distinto
#         hue="Modelo",          # Diferenciar por modelo con colores
#         split=True,            
#         inner="box",             # Mostrar caja con mediana y cuartiles
#         density_norm="width",
#         col_wrap=3,              # Máximo 3 paneles por fila
#         height=4,
#     )

#     g.set_titles("{col_name}")
#     g.set_axis_labels("Categorías", "Puntuación")
#     g.set_xticklabels(rotation=15, ha="right")
#     g.fig.suptitle("Distribución de Puntuaciones por Categoría y Modelo (Violinplot)", y=1.03)
#     g.tight_layout()
#     return g.fig


def plot_violin_llm_por_prompt():
    """
    Genera un violinplot mostrando la distribución de las puntuaciones (A–E)
    para cada modelo y prompt, sin modificar 'matriz_resultados'.
    """
    # Crear una copia en formato largo (sin alterar el original)
    df_long = matriz_resultados.reset_index().melt(
        id_vars=["Modelo", "Prompt"],
        value_vars=["A", "B", "C", "D", "E"],
        var_name="Categoria",
        value_name="Puntuacion"
    )

    colores = ["#4C72B0", "#DD8452", "#55A868","#6A8CC7", "#E59C6D", "#7CC79A"]

    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=df_long,
        x="Modelo",
        y="Puntuacion",
        hue="Prompt",
        palette=colores,
        split=True,
        inner="box",
        density_norm="width",
        cut=0,
        width=0.8,
        gap=0.2
    )

    # Líneas verticales entre modelos
    modelos = df_long["Modelo"].unique()
    for i in range(len(modelos) - 1):
        ax.axvline(x=i + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.title("Density distribution of scores for the Large Language Model", pad=40)
    plt.xlabel("Large Language Model")
    plt.ylabel("Score")
    
    #plt.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc="upper left")
    # Obtener handles y labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()  # Quitar la leyenda automática

    # Crear leyenda horizontal centrada
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        handletextpad=0.5
    )

    fig = plt.gcf()
    fig.text(0.23, 0.85, "Prompt:", ha="right", va="center", fontsize=10)

    # Ajuste de espacio superior
    plt.subplots_adjust(top=0.82)
    return plt.gcf()



plot_functions = [
    plot_heatmap_prompt1_vs_expertos,
    plot_heatmap_prompt2_vs_expertos,
    plot_heatmap_prompt3_vs_expertos,
    plot_heatmap_promedios_generales,
    plot_heatmap_diferencias
    ]

handle_plots(plot_functions, show=True, save=False, prefix="analisis_resultados")