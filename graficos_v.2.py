import re
import pandas as pd
import numpy as np
import seaborn as sns
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


# ----------------- Cargar y procesar datos -----------------
xls = pd.read_excel("Resultados LLM.xlsx", sheet_name=None)
excluded = {"Transcripciones"}
sheet_names = [name for name in xls.keys() if name not in excluded]

if "Expertos" not in xls:
    raise ValueError("No se encontró la hoja 'Expertos' en el archivo.")
df_expertos = xls["Expertos"]
promedios_expertos = df_expertos[["A","B","C","D","E"]].mean(axis=0)

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

frames = []
skipped_sheets = []
parsed_info = []
prompts_seen = set()
json_seen = set()

for sheet in sheet_names:
    m = pattern.match(sheet)
    if not m:
        skipped_sheets.append(sheet)
        continue

    model = m.group("model").strip()
    prompt_num = m.group("prompt")
    json_flag = bool(m.group("json"))

    prompts_seen.add(prompt_num)
    json_seen.add("json" if json_flag else "no_json")

    df_sheet = xls[sheet]
    prom1 = process_sheet(df_sheet)
    if prom1 is None:
        skipped_sheets.append(sheet)
        continue

    df_temp = pd.DataFrame({
        "Categoria": prom1.index,
        "Promedio": prom1.values,
        "Expertos": promedios_expertos.reindex(prom1.index).values,
        "LLM": model,
        "Prompt": prompt_num,
        "JSON": "json" if json_flag else "no_json"
    })
    frames.append(df_temp)
    parsed_info.append((sheet, model, prompt_num, json_flag))

# Añadir Expertos replicado
ordered_cats = ["A","B","C","D","E"]
proms = promedios_expertos.reindex(ordered_cats).fillna(np.nan)
if not prompts_seen:
    prompts_seen = {"0"}
if not json_seen:
    json_seen = {"no_json"}

for p in sorted(prompts_seen, key=lambda x: int(x) if str(x).isdigit() else x):
    for j in sorted(json_seen):
        df_experts_rep = pd.DataFrame({
            "Categoria": proms.index,
            "Promedio": proms.values,
            "Expertos": proms.values,
            "LLM": "Expertos",
            "Prompt": p,
            "JSON": j
        })
        frames.append(df_experts_rep)
parsed_info.append(("Expertos_replicado", "Expertos", ",".join(sorted(prompts_seen)), ",".join(sorted(json_seen))))

df_final = pd.concat(frames, ignore_index=True).dropna(subset=["Promedio"]) if frames else pd.DataFrame(columns=["Categoria","Promedio","Expertos","LLM","Prompt","JSON"])
df_final["Categoria"] = pd.Categorical(df_final["Categoria"], categories=ordered_cats, ordered=True)

def safe_int_convert(col):
    try:
        return col.astype(int)
    except:
        return pd.Categorical(col).codes + 1

df_final["Prompt_orig"] = df_final["Prompt"]
df_final["Prompt"] = safe_int_convert(df_final["Prompt"].astype(str))
df_final["JSON"] = df_final["JSON"].fillna("no_json")

# ----------------- Funciones para crear gráficos ---ARIEL SOLO AGREGA LOS GRAFICOS AQUI -----------------
def grafico1():
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
    g.fig.suptitle("Promedio por Categoría — Prompt (col) y JSON (row)")
    g.set(ylim=(0,5))
    return g.fig

def grafico2():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_final,
        x="Promedio", y="Categoria",
        hue="LLM", style="JSON",
        size="Prompt", sizes=(50, 300),
        alpha=0.9,
        ax=ax
    )
    ax.set_title("Promedio vs Categoria — color: LLM | estilo: JSON | tamaño: Prompt")
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    return fig

def grafico3():
    df_agg = df_final.groupby(["LLM","Categoria"], as_index=False, observed=True)["Promedio"].mean()
    g = sns.catplot(
        data=df_agg,
        x="Categoria", y="Promedio", hue="LLM",
        kind="point", dodge=True, height=5, aspect=1.4,
        order=ordered_cats
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Comparación agregada: Promedio por Categoría — LLMs (incluye Expertos)")
    g.set(ylim=(0,5))
    return g.fig

# ----------------- Ejecutar todo ARIEL PON EL NOMBRE DE LA FUNCION AQUI-----------------
plot_funcs = [grafico1, grafico2, grafico3]
# save guarda el plot en png si True
handle_plots(plot_funcs, show=True, save=False, prefix="grafico")
