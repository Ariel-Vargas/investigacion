import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import numpy as np

prompt_ChatGPT = pd.read_excel('Resultados LLM.xlsx', sheet_name="ChatGPT")
prompt_Gemini = pd.read_excel('Resultados LLM.xlsx', sheet_name="Gemini")
prompt_Deepseek = pd.read_excel('Resultados LLM.xlsx', sheet_name="Deepseek")

prompt_LLama = pd.read_excel('Resultados LLM.xlsx', sheet_name="Llama")
prompt_Claude = pd.read_excel('Resultados LLM.xlsx', sheet_name="Claude")
prompt_Grok = pd.read_excel('Resultados LLM.xlsx', sheet_name="Grok")

grade_expertos = pd.read_excel('Resultados LLM.xlsx', sheet_name="Expertos")
promedios_expertos = np.mean(grade_expertos[["A","B","C","D","E"]], axis=0)
#print(promedios_expertos)


prompt1 = prompt_ChatGPT.iloc[:, 0:6]
prompt2 = prompt_ChatGPT.loc[:, ["ID","A.1","B.1","C.1","D.1","E.1"]]
prompt2.rename(columns = {"A.1":'A', "B.1":"B", "C.1":"C","D.1":"D","E.1":"E"}, inplace = True)

promedios_p1 = np.mean(prompt1[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 1:\n", promedios_p1)

promedios_p2 = np.mean(prompt2[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 2:\n", promedios_p2)


prompt1_gem = prompt_Gemini.iloc[:, 0:6]
prompt2_gem = prompt_Gemini.loc[:, ["ID","A.1","B.1","C.1","D.1","E.1"]]
prompt2_gem.rename(columns = {"A.1":'A', "B.1":"B", "C.1":"C","D.1":"D","E.1":"E"}, inplace = True)
promedios_p1_gem = np.mean(prompt1_gem[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 1 Gemini:\n", promedios_p1_gem)
promedios_p2_gem = np.mean(prompt2_gem[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 2 Gemini:\n", promedios_p2_gem)

prompt1_deep = prompt_Deepseek.iloc[:, 0:6]
prompt2_deep = prompt_Deepseek.loc[:, ["ID","A.1","B.1","C.1","D.1","E.1"]]
prompt2_deep.rename(columns = {"A.1":'A', "B.1":"B", "C.1":"C","D.1":"D","E.1":"E"}, inplace = True)
promedios_p1_deep = np.mean(prompt1_deep[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 1 Deepseek:\n", promedios_p1_deep)
promedios_p2_deep = np.mean(prompt2_deep[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 2 Deepseek:\n", promedios_p2_deep)

p1_Llama = prompt_LLama.iloc[:, 0:6]
p2_Llama = prompt_LLama.loc[:, ["ID","A.1","B.1","C.1","D.1","E.1"]]
p2_Llama.rename(columns = {"A.1":'A', "B.1":"B", "C.1":"C","D.1":"D","E.1":"E"}, inplace = True)
promedios_p1_Llama = np.mean(p1_Llama[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 1 Llama:\n", promedios_p1_Llama)
promedios_p2_Llama = np.mean(p2_Llama[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 2 Llama:\n", promedios_p2_Llama)

p1_Claude = prompt_Claude.iloc[:, 0:6]
p2_Claude = prompt_Claude.loc[:, ["ID","A.1","B.1","C.1","D.1","E.1"]]
p2_Claude.rename(columns = {"A.1":'A', "B.1":"B", "C.1":"C","D.1":"D","E.1":"E"}, inplace = True)
promedios_p1_Claude = np.mean(p1_Claude[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 1 Claude:\n", promedios_p1_Claude)
promedios_p2_Claude = np.mean(p2_Claude[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 2 Claude:\n", promedios_p2_Claude)

p1_Grok = prompt_Grok.iloc[:, 0:6]
p2_Grok = prompt_Grok.loc[:, ["ID","A.1","B.1","C.1","D.1","E.1"]]
p2_Grok.rename(columns = {"A.1":'A', "B.1":"B", "C.1":"C","D.1":"D","E.1":"E"}, inplace = True)
promedios_p1_Grok = np.mean(p1_Grok[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 1 Grok:\n", promedios_p1_Grok)
promedios_p2_Grok = np.mean(p2_Grok[["A","B","C","D","E"]], axis=0)
#print("Promedios prompt 2 Grok:\n", promedios_p2_Grok)


df_gpt = pd.DataFrame({
    "Categoria": promedios_p1.index,
    "Promedio": promedios_p1.values,
    "Expertos": promedios_expertos.values,
    "LLM": "ChatGPT",
    "Prompt": 1
})
df_gpt2 = pd.DataFrame({
    "Categoria": promedios_p2.index,
    "Promedio": promedios_p2.values,
    "Expertos": promedios_expertos.values,
    "LLM": "ChatGPT",
    "Prompt": 2
})

df_chatgpt = pd.concat([df_gpt, df_gpt2], ignore_index=True)


df_gem = pd.DataFrame({
    "Categoria": promedios_p1_gem.index,
    "Promedio": promedios_p1_gem.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Gemini",
    "Prompt": 1
})
df_gem2 = pd.DataFrame({
    "Categoria": promedios_p2_gem.index,
    "Promedio": promedios_p2_gem.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Gemini",
    "Prompt": 2
})
df_gemini = pd.concat([df_gem, df_gem2], ignore_index=True)

df_deep = pd.DataFrame({
    "Categoria": promedios_p1_deep.index,
    "Promedio": promedios_p1_deep.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Deepseek",
    "Prompt": 1
})
df_deep2 = pd.DataFrame({
    "Categoria": promedios_p2_deep.index,
    "Promedio": promedios_p2_deep.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Deepseek",
    "Prompt": 2
})
df_deepseek = pd.concat([df_deep, df_deep2], ignore_index=True)

df_LLama1 = pd.DataFrame({
    "Categoria": promedios_p1_Llama.index,
    "Promedio": promedios_p1_Llama.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Llama",
    "Prompt": 1
})
df_LLama2 = pd.DataFrame({
    "Categoria": promedios_p2_Llama.index,
    "Promedio": promedios_p2_Llama.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Llama",
    "Prompt": 2
})
df_Llama = pd.concat([df_LLama1, df_LLama2], ignore_index=True)

df_Claude1 = pd.DataFrame({
    "Categoria": promedios_p1_Claude.index,
    "Promedio": promedios_p1_Claude.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Claude",
    "Prompt": 1
})
df_Claude2 = pd.DataFrame({
    "Categoria": promedios_p2_Claude.index,
    "Promedio": promedios_p2_Claude.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Claude",
    "Prompt": 2
})
df_Claude = pd.concat([df_Claude1, df_Claude2], ignore_index=True)

df_Grok1 = pd.DataFrame({
    "Categoria": promedios_p1_Grok.index,
    "Promedio": promedios_p1_Grok.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Grok",
    "Prompt": 1
})
df_Grok2 = pd.DataFrame({
    "Categoria": promedios_p2_Grok.index,
    "Promedio": promedios_p2_Grok.values,
    "Expertos": promedios_expertos.values,
    "LLM": "Grok",
    "Prompt": 2
})
df_Grok = pd.concat([df_Grok1, df_Grok2], ignore_index=True)

df_final = pd.concat([df_chatgpt, df_gemini, df_deepseek, df_Llama, df_Claude, df_Grok], ignore_index=True)

print(df_final)
print(df_final["Promedio"].mean())
print(df_final["Expertos"].mean())



# sns.scatterplot(data=df_final, x="Categoria", y="Promedio", hue="LLM", style="Prompt", s=100)
# plt.title("Comparación de Promedios por Categoría - LLM y Prompt")
# plt.show()

# sns.scatterplot(data=df_final, x="Promedio", y="Categoria", hue="LLM", style="Prompt", s=100)
# plt.title("Comparación de Promedios por Categoría - LLM y Prompt")
# plt.show()

# sns.lineplot(data=df_final, x="Categoria", y="Promedio", hue="LLM", style="Prompt", markers=True, dashes=False)
# plt.title("Comparación de Promedios por Categoría - LLM y Prompt") 
# plt.show()

# sns.catplot(data=df_final, x="Categoria", y="Promedio", hue="LLM", col="Prompt", kind="point", height=5, aspect=1)
# plt.suptitle("Catplot Comparación de Promedios por Categoría - LLM y Prompt", y=1.02)
# plt.show()


##### Desde aquí ####

#plt.figure(figsize=(12,6))
# sns.boxplot(
#     data=df_final,
#     x="Categoria", y="Promedio",
#     hue="LLM",
# )
# #plt.ylim(0, 5)
# plt.show()

# sns.jointplot(
#     data=df_final,
#     x="Expertos", y="Promedio",
#     hue="LLM",
#     kind="scatter",
#     height=8,
#     ratio=5,
#     marginal_ticks=True,
#     alpha=0.7
# )
# #plt.ylim(0, 5)
# plt.show()

# plt.figure(figsize=(10,6))
# sns.scatterplot(
#     data=df_final,
#     x="Promedio", y="Expertos",
#     hue="LLM", style="Prompt", s=100
# )
# plt.title("Relación Promedio vs Expertos por LLM y Prompt")
# plt.show()

sns.pairplot(
    df_final,
    vars=["Promedio", "Expertos"],
    hue="Prompt",
    diag_kind="kde"
)
plt.show()


