import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import numpy as np

prompt1_ChatGPT = pd.read_excel('Resultados LLM.xlsx', sheet_name="ChatGPT")

prompt1_Gemini = pd.read_excel('Resultados LLM.xlsx', sheet_name="Gemini")

prompt1_Deepseek = pd.read_excel('Resultados LLM.xlsx', sheet_name="Deepseek")

prompt1 = prompt1_ChatGPT.iloc[:, 0:7]
prompt2 = prompt1_ChatGPT.loc[:, ["ID","A.1","B.1","C.1","D.1","E.1","Total.1"]]
prompt2.rename(columns = {"A.1":'A', "B.1":"B", "C.1":"C","D.1":"D","E.1":"E","Total.1":"Total"}, inplace = True)




# prompt1_long = prompt2.melt(id_vars=["ID"], value_vars=["A","B","C","D","E"], var_name="Categoria", value_name="Puntaje")

# print(prompt1_long)




varianzas = prompt1[["A","B","C","D","E","Total"]].var()
print("Varianzas:\n", varianzas)

# Calcular medias y desviaciones
medias = prompt1[["A","B","C","D","E","Total"]].mean()
desv = prompt1[["A","B","C","D","E","Total"]].std()

# Umbral: rango de aceptación
umbral_inf = medias - desv
umbral_sup = medias + desv

print("\nUmbral inferior:\n", umbral_inf)
print("\nUmbral superior:\n", umbral_sup)

resultados = pd.DataFrame(index=prompt1.index)

for col in ["A","B","C","D","E","Total"]:
    resultados[col] = prompt1[col].between(umbral_inf[col], umbral_sup[col])

# Añadir ID
resultados["ID"] = prompt1["ID"]

print("\nResultados de aceptación por ID:\n")
print(resultados)

resultados["Aceptado"] = resultados[["A","B","C","D","E","Total"]].all(axis=1)
print("\nResumen por ID:\n", resultados[["ID","Aceptado"]])




# Gráfico radar comparativo

# categorias = ["A","B","C","D","E"]

# def preparar_datos(df):
#     return df[categorias + ["ID"]].copy()

# chat = preparar_datos(prompt1_ChatGPT)
# gemini = preparar_datos(prompt1_Gemini)
# deepseek = preparar_datos(prompt1_Deepseek)

# # Elegir un ID para graficar
# id_graf = 5  # primer registro
# valores_chat = chat.loc[id_graf, categorias].tolist()
# valores_gemini = gemini.loc[id_graf, categorias].tolist()
# valores_deepseek = deepseek.loc[id_graf, categorias].tolist()

# # Cerrar el círculo para radar
# valores_chat += valores_chat[:1]
# valores_gemini += valores_gemini[:1]
# valores_deepseek += valores_deepseek[:1]

# # Ángulos de cada categoría
# num_vars = len(categorias)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]

# # Crear gráfico
# fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

# ax.plot(angles, valores_chat, color='red', linewidth=2, label='ChatGPT')
# ax.fill(angles, valores_chat, color='red', alpha=0.25)

# ax.plot(angles, valores_gemini, color='blue', linewidth=2, label='Gemini')
# ax.fill(angles, valores_gemini, color='blue', alpha=0.25)

# ax.plot(angles, valores_deepseek, color='green', linewidth=2, label='Deepseek')
# ax.fill(angles, valores_deepseek, color='green', alpha=0.25)

# # Configurar etiquetas y límites
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categorias)
# ax.set_ylim(0, max(chat[categorias].max().max(), gemini[categorias].max().max(), deepseek[categorias].max().max()) + 5)

# plt.title(f"Radar Chart comparativo por categoría y Total - ID {chat.loc[id_graf,'ID']}")
# plt.legend(loc='upper right')
# plt.show()



# Gráfico de barras
# sns.barplot(data=prompt1_long, x="ID", y="Puntaje", hue="Categoria")
# plt.xticks(rotation=90)
# plt.show()

# print(prompt1)
# print("")
# print(prompt2)



