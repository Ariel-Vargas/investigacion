import re
import pandas as pd  # type: ignore
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
import statsmodels.api as sm # type: ignore
from statsmodels.formula.api import ols # type: ignore
import pingouin as pg  # type: ignore # Para ICC


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
excluded = {"Transcripciones", "Expertos"}
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
print(df_final.head(n=25))

print("\nEstadísticas generales:")
print("Media Promedio:", df_final["Promedio"].mean())
print("Media Expertos:", df_final["Expertos"].mean())



def safe_int_convert(col):
    try:
        return col.astype(int)
    except:
        return pd.Categorical(col).codes + 1

df_final["Prompt_orig"] = df_final["Prompt"]
df_final["Prompt"] = safe_int_convert(df_final["Prompt"].astype(str))
df_final["JSON"] = df_final["JSON"].fillna("no_json")

# ----------------- Funciones para crear gráficos -----------------
# ----------------- FUNCIONES DE BOXPLOTS -----------------
def boxplot_general():
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(
        data=df_final,
        x="Categoria", y="Promedio",
        hue="LLM",
        ax=ax
    )
    ax.set_title("Distribución de Promedios por Categoría y LLM")
    ax.set_ylim(0,5)
    return fig


def boxplot_por_llm(llm_name, palette=None):
    df_llm = df_final[df_final["LLM"] == llm_name]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(
        data=df_llm,
        x="Categoria", y="Promedio",
        hue="Prompt",
        palette=palette or ["#1f77b4", "#ff7f0e", "#2ca02c"],
        ax=ax
    )
    # Líneas divisorias entre categorías
    for i in range(len(df_final["Categoria"].unique()) - 1):
        ax.axvline(i + 0.5, color="gray", linestyle="--", alpha=0.7)

    ax.set_title(f"Distribución de Promedios por Categoría — {llm_name} (Prompts)")
    #ax.set_ylim(0,5)
    return fig


def boxplot_facetas():
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
    #g.set(ylim=(0,5))
    return g.fig


# ----------------- FUNCIONES DE VIOLINPLOTS -----------------

def violinplot_general():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df_final,
        x="Categoria", y="Promedio",
        hue="LLM",
        split=False,     # no dividir violines por hue
        density_norm="width",  # escala para que todos los violines tengan el mismo ancho
        inner="quartile", # muestra la mediana y cuartiles
        cut=0,           # no extrapola más allá de los datos
        ax=ax
    )
    ax.set_title("Distribución de densidad (Violin Plot) por Categoría y LLM")
    ax.set_ylim(0, 5)
    return fig


def flechas_global_json():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Asegurar tipo numérico
    df_final["Promedio"] = pd.to_numeric(df_final["Promedio"], errors="coerce")
    df_final["Expertos"] = pd.to_numeric(df_final["Expertos"], errors="coerce")

    # Agrupar por Prompt y JSON → promedio global (json y no_json)
    df_cat = df_final.groupby(["JSON", "Prompt"], as_index=False)["Promedio"].mean()

    # Agrupar por Prompt para Expertos
    df_expertos = df_final.groupby("Prompt", as_index=False)["Expertos"].mean()
    df_expertos["JSON"] = "expertos"   # le damos la etiqueta de categoría
    df_expertos = df_expertos.rename(columns={"Expertos": "Promedio"})  # para que coincida

    # Unir todo en un solo DataFrame
    df_cat = pd.concat([df_cat, df_expertos], ignore_index=True)
    df_cat["Prompt"] = df_cat["Prompt"].astype(str)

    print(df_cat)  # <-- así verás json, no_json y expertos

    # Colores fijos para JSON y noJSON
    colores = {"json": "tab:blue", "no_json": "tab:orange", "expertos": "tab:green"}

    for json_flag, estilo in [("json", "dashed"), ("no_json", "dashed"), ("expertos", "dashed")]:
        subjson = df_cat[df_cat["JSON"] == json_flag]

        if not subjson.empty and set(subjson["Prompt"]) >= {"1", "2", "3"}:
            # Ordenar por prompt
            subjson = subjson.sort_values("Prompt")

            x_vals = subjson["Prompt"].astype(int).values
            y_vals = subjson["Promedio"].values

            # Dibujar línea con flechas
            ax.plot(x_vals, y_vals, marker="o",
                    color=colores[json_flag], linestyle=estilo,
                    label=f"{json_flag.upper()}")

            # Flechas entre puntos consecutivos
            for j in range(len(x_vals) - 1):
                ax.annotate("",
                    xy=(x_vals[j+1], y_vals[j+1]),
                    xytext=(x_vals[j], y_vals[j]),
                    arrowprops=dict(arrowstyle="->", color=colores[json_flag], lw=1.5)
                )

    # Etiquetas
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Promedio Global (todos los LLM, todas las categorías)")
    ax.set_title("Evolución Global JSON vs NoJSON")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Prompt 1", "Prompt 2", "Prompt 3"])
    ax.grid(alpha=0.3)
    ax.legend()

    return fig


# ----------------- FUNCIONES DE CATPLOTS CON FLECHAS POR LLM -----------------
def flechas_llm_json(llm_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filtrar por un LLM específico
    df_llm = df_final[df_final["LLM"] == llm_name].copy()
    df_llm["Promedio"] = pd.to_numeric(df_llm["Promedio"], errors="coerce")

    # Agrupar solo por JSON y Prompt
    df_cat = df_llm.groupby(["JSON", "Prompt"], as_index=False)["Promedio"].mean()
    df_cat["Prompt"] = df_cat["Prompt"].astype(str)

    # Colores fijos para JSON y noJSON
    colores = {"json": "tab:blue", "no_json": "tab:orange"}

    for json_flag, estilo in [("json", "solid"), ("no_json", "dashed")]:
        subjson = df_cat[df_cat["JSON"] == json_flag]

        if set(subjson["Prompt"]) >= {"1", "2", "3"}:
            # Ordenar por prompt
            subjson = subjson.sort_values("Prompt")

            x_vals = subjson["Prompt"].astype(int).values
            y_vals = subjson["Promedio"].values

            # Dibujar línea con flechas
            ax.plot(x_vals, y_vals, marker="o",
                    color=colores[json_flag], linestyle=estilo,
                    label=f"{json_flag.upper()}")

            # Flechas entre puntos consecutivos
            for j in range(len(x_vals) - 1):
                ax.annotate("",
                    xy=(x_vals[j+1], y_vals[j+1]),
                    xytext=(x_vals[j], y_vals[j]),
                    arrowprops=dict(arrowstyle="->", color=colores[json_flag], lw=1.5)
                )

    # Etiquetas
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Promedio")
    ax.set_title(f"Evolución JSON vs NoJSON en {llm_name}")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Prompt 1", "Prompt 2", "Prompt 3"])
    ax.grid(alpha=0.3)
    ax.legend()

    return fig

#----------------- FUNCION DE CATPLOTS CON FLECHAS DE TODOS LOS LLM  -----------------

def catplot_flechas_llm_json_expertos():
    # Asegurar tipo numérico
    df_final["Promedio"] = pd.to_numeric(df_final["Promedio"], errors="coerce")
    df_final["Expertos"] = pd.to_numeric(df_final["Expertos"], errors="coerce")
    df_final["Prompt"] = df_final["Prompt"].astype(int)

    # Crear DataFrame combinado
    df_expertos = df_final[["LLM", "Prompt", "Expertos"]].copy()
    df_expertos.rename(columns={"Expertos": "Promedio"}, inplace=True)
    df_expertos["JSON"] = "expertos"  # Marcar como expertos

    # Concatenar con los JSON y noJSON
    df_cat = pd.concat([
        df_final[["LLM", "Prompt", "Promedio", "JSON"]],
        df_expertos
    ], ignore_index=True)
    print(df_cat)

    # Colores fijos
    colores = {"json": "tab:blue", "no_json": "tab:orange", "expertos": "tab:green"}

    # Crear catplot
    g = sns.catplot(
        data=df_cat,
        x="Prompt", y="Promedio",
        hue="JSON", col="LLM",
        kind="point",
        col_wrap=3,
        dodge=False,
        height=4, aspect=1.2,
        markers="o", linestyles="dashed"
    )

    # Iterar sobre cada subplot y dibujar flechas
    for ax, (llm_name, subdf) in zip(g.axes.flat, df_cat.groupby("LLM")):

        for json_flag, estilo in [("json", "dashed"), ("no_json", "dashed"), ("expertos", "dashed")]:
            subjson = subdf[subdf["JSON"] == json_flag]

            if len(subjson) > 1:
                # Ordenar por Prompt
                subjson = subjson.sort_values("Prompt")

                # Usar el valor exacto de Prompt como X (no dodge)
                line = ax.lines[-1]  # Última línea agregada de seaborn pointplot
                x_vals = line.get_xdata()
                y_vals = line.get_ydata()


                # Dibujar línea con marcador
                ax.plot(x_vals, y_vals, marker="o",
                        color=colores[json_flag], linestyle=estilo,
                        label=f"{json_flag.upper()}")

                # Dibujar flechas exactas entre prompts consecutivos
                for j in range(len(x_vals)-1):
                    ax.annotate("",
                        xy=(x_vals[j+1], y_vals[j+1]),
                        xytext=(x_vals[j], y_vals[j]),
                        arrowprops=dict(arrowstyle="->", color=colores[json_flag], lw=1.5))

    # Ajustes de títulos
    g.set_titles("{col_name}")
    g.set_axis_labels("Prompt", "Promedio")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Evolución JSON, NoJSON y Expertos por LLM", fontsize=14)

    return g.fig

def flechas_global_llm_json(llm_list):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filtrar solo los LLM seleccionados
    df_filtered = df_final[df_final["LLM"].isin(llm_list)].copy()

    # Asegurar tipo numérico
    df_filtered["Promedio"] = pd.to_numeric(df_filtered["Promedio"], errors="coerce")
    df_filtered["Expertos"] = pd.to_numeric(df_filtered["Expertos"], errors="coerce")

    # Agrupar por Prompt y JSON → promedio global de json/no_json
    df_cat = df_filtered.groupby(["JSON", "Prompt"], as_index=False)["Promedio"].mean()

    # Agrupar por Prompt para Expertos
    df_expertos = df_filtered.groupby("Prompt", as_index=False)["Expertos"].mean()
    df_expertos["JSON"] = "expertos"
    df_expertos = df_expertos.rename(columns={"Expertos": "Promedio"})

    # Unir todo en un solo DataFrame
    df_cat = pd.concat([df_cat, df_expertos], ignore_index=True)
    df_cat["Prompt"] = df_cat["Prompt"].astype(str)

    # Colores
    colores = {"json": "tab:blue", "no_json": "tab:orange", "expertos": "tab:green"}

    markers = ["o", "s", "^"]

    for json_flag, estilo in [("json", "dashed"), ("no_json", "dashed"), ("expertos", "dashed")]:
        subjson = df_cat[df_cat["JSON"] == json_flag]

        if not subjson.empty and set(subjson["Prompt"]) >= {"1", "2", "3"}:
            subjson = subjson.sort_values("Prompt")

            x_vals = subjson["Prompt"].astype(int).values
            y_vals = subjson["Promedio"].values

            # Línea
            ax.plot(x_vals, y_vals,
                    color=colores[json_flag], linestyle=estilo,
                    label=f"{json_flag.upper()}")

            # Poner markers distintos en cada punto
            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                marker = markers[i % len(markers)]  # cicla si hay más de 3 prompts
                ax.plot(x, y, marker=marker, color=colores[json_flag])

            # Flechas
            for j in range(len(x_vals) - 1):
                ax.annotate("",
                    xy=(x_vals[j+1], y_vals[j+1]),
                    xytext=(x_vals[j], y_vals[j]),
                    arrowprops=dict(arrowstyle="->", color=colores[json_flag], lw=1.5)
                )

    # Etiquetas
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Promedio Global")
    ax.set_title(f"Evolución Global JSON vs NoJSON vs Expertos\nLLM: {', '.join(llm_list)}")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Prompt 1", "Prompt 2", "Prompt 3"])
    ax.grid(alpha=0.3)
    ax.legend()

    return fig



# ----------------- Ejecutar todo -----------------
plot_funcs = [
    #flechas_global_json,
    #lambda: flechas_global_llm_json(["Grok", "Gemini"]),
    #lambda: flechas_llm_json("ChatGPT"),
    catplot_flechas_llm_json_expertos, # gráfico con facetas por LLM y flechas cada uno + expertos
    # boxplot_general,
    # violinplot_general,
    # lambda: boxplot_por_llm("ChatGPT"),
    # lambda: boxplot_por_llm("Gemini"),
    # lambda: boxplot_por_llm("Deepseek"),
    # lambda: boxplot_por_llm("Meta"),
    # lambda: boxplot_por_llm("Claude"),
    # lambda: boxplot_por_llm("Grok"),
    #boxplot_facetas
]
# save guarda el plot en png si True
handle_plots(plot_funcs, show=True, save=False, prefix="grafico")


# ----------------- ANOVA -----------------
# ANOVA: Comparar Promedio según LLM
# anova_model = ols("Promedio ~ C(LLM)", data=df_final).fit()
# anova_table = sm.stats.anova_lm(anova_model, typ=2)
# print("=== ANOVA: Diferencias entre LLM ===")
# print(anova_table)

# # ANOVA vs Expertos: comparar cada LLM con la media de expertos
# df_anova_exp = df_final.copy()
# df_anova_exp["LLM_vs_Expertos"] = df_anova_exp["LLM"] + "_vs_Expertos"

# # Crear una fila por experto para cada categoría
# df_melt = pd.melt(df_anova_exp, id_vars=["Categoria", "LLM"], value_vars=["Promedio", "Expertos"],
#                   var_name="Fuente", value_name="Valor")
# anova_model_exp = ols("Valor ~ C(LLM) + C(Fuente)", data=df_melt).fit()
# anova_table_exp = sm.stats.anova_lm(anova_model_exp, typ=2)
# print("\n=== ANOVA: LLM vs Expertos ===")
# print(anova_table_exp)

# # ----------------- CORRELACIÓN INTRACLASE (ICC) -----------------
# # ICC tipo (2,1): dos efectos aleatorios, consistencia entre LLM y Expertos
# icc_list = []
# categorias = df_final["Categoria"].unique()

# for cat in categorias:
#     df_cat = df_final[df_final["Categoria"] == cat].copy()
#     # Reorganizar para pingouin
#     df_icc = df_cat[["LLM", "Promedio", "Expertos"]].copy()
#     # Transformar a formato largo
#     df_icc_long = pd.melt(df_icc, id_vars="LLM", value_vars=["Promedio", "Expertos"],
#                           var_name="Rater", value_name="Score")
#     icc_res = pg.intraclass_corr(data=df_icc_long, targets="LLM", raters="Rater", ratings="Score")
#     icc_val = icc_res.loc[icc_res["Type"]=="ICC2", "ICC"].values[0]
#     icc_list.append({"Categoria": cat, "ICC2": icc_val})

# icc_table = pd.DataFrame(icc_list)
# print("\n=== Correlación Intraclase (ICC2) entre LLM y Expertos por Categoría ===")
# print(icc_table)


# df_chatgpt = df_final[df_final["LLM"] == "ChatGPT"].copy()

# # ----------------- ANOVA ChatGPT vs Expertos -----------------
# # Preparar DataFrame largo para ANOVA
# df_melt = pd.melt(df_chatgpt, id_vars=["Categoria"], value_vars=["Promedio", "Expertos"],
#                   var_name="Fuente", value_name="Valor")

# anova_model_chat = ols("Valor ~ C(Fuente)", data=df_melt).fit()
# anova_table_chat = sm.stats.anova_lm(anova_model_chat, typ=2)

# print("=== ANOVA: ChatGPT vs Expertos ===")
# print(anova_table_chat)

# # ----------------- ICC ChatGPT vs Expertos -----------------
# # Preparar DataFrame largo para ICC
# df_icc_long = pd.melt(df_chatgpt, id_vars=["Categoria"], value_vars=["Promedio", "Expertos"],
#                       var_name="Rater", value_name="Score")

# icc_res = pg.intraclass_corr(data=df_icc_long, targets="Categoria", raters="Rater", ratings="Score")
# icc_val = icc_res.loc[icc_res["Type"]=="ICC2", ["ICC", "CI95%"]]

# print("\n=== ICC2: ChatGPT vs Expertos por Categoría ===")
# print(icc_val)

# ----------------- ANOVA por Prompt -----------------
# prompts = sorted(df_final["Prompt"].unique())
# anova_results = {}
# icc_results = {}

# for p in prompts:
#     df_prompt = df_final[df_final["Prompt"] == p].copy()
    
#     # --- ANOVA LLM vs Expertos ---
#     df_melt = pd.melt(df_prompt, id_vars=["Categoria"], value_vars=["Promedio", "Expertos"],
#                       var_name="Fuente", value_name="Valor")
    
#     anova_model = ols("Valor ~ C(Fuente)", data=df_melt).fit()
#     anova_table = sm.stats.anova_lm(anova_model, typ=2)
#     anova_results[p] = anova_table
    
#     # --- ICC LLM vs Expertos ---
#     df_icc_long = pd.melt(df_prompt, id_vars=["Categoria"], value_vars=["Promedio", "Expertos"],
#                           var_name="Rater", value_name="Score")
#     icc_res = pg.intraclass_corr(data=df_icc_long, targets="Categoria", raters="Rater", ratings="Score")
#     icc_val = icc_res.loc[icc_res["Type"]=="ICC2", ["ICC", "CI95%"]]
#     icc_results[p] = icc_val

# # ----------------- Mostrar resultados -----------------
# for p in prompts:
#     print(f"\n=== ANOVA: LLM vs Expertos — Prompt {p} ===")
#     print(anova_results[p])
    
#     print(f"\n=== ICC2: LLM vs Expertos — Prompt {p} ===")
#     print(icc_results[p])


# ----------------- ANOVA + ICC por cada LLM -----------------
# llms = df_final["LLM"].unique()
# anova_results = {}
# icc_results = {}

# for llm in llms:
#     df_llm = df_final[df_final["LLM"] == llm].copy()
    
#     # --- ANOVA: LLM vs Expertos ---
#     df_melt = pd.melt(df_llm, id_vars=["Categoria"], value_vars=["Promedio", "Expertos"],
#                       var_name="Fuente", value_name="Valor")
    
#     anova_model = ols("Valor ~ C(Fuente)", data=df_melt).fit()
#     anova_table = sm.stats.anova_lm(anova_model, typ=2)
#     anova_results[llm] = anova_table
    
#     # --- ICC2: LLM vs Expertos ---
#     df_icc_long = pd.melt(df_llm, id_vars=["Categoria"], value_vars=["Promedio", "Expertos"],
#                           var_name="Rater", value_name="Score")
    
#     icc_res = pg.intraclass_corr(data=df_icc_long,
#                                  targets="Categoria", raters="Rater", ratings="Score")
    
#     icc_val = icc_res.loc[icc_res["Type"] == "ICC2", ["ICC", "CI95%"]]
#     icc_results[llm] = icc_val

# # ----------------- Mostrar resultados -----------------
# for llm in llms:
#     print(f"\n=== ANOVA: {llm} vs Expertos ===")
#     print(anova_results[llm])
    
#     print(f"\n=== ICC2: {llm} vs Expertos ===")
#     print(icc_results[llm])