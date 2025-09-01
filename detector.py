import glob
import json
import time
from playwright.sync_api import sync_playwright # type: ignore


transcripciones_folder = glob.glob('transcripciones\*.json')
prompts_folder = glob.glob('prompts\*.txt')

resultados = []
for archivo in transcripciones_folder:
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = json.load(f)
        for id, transcripcion in contenido["entries"].items():
            resultados.append({
                "id": id,
                "transcripcion": transcripcion,
                "resultados": {}
            })


# === Funciones de evaluación ===
def evaluar_chatgpt(playwright, texto, prompt):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(storage_state="chatgpt_state.json")  # cookies guardadas
    page = context.new_page()
    page.goto("https://chat.openai.com/")

    page.wait_for_timeout(3000)
    page.fill('textarea', f"{prompt}\n\n{texto}")
    page.keyboard.press("Enter")

    page.wait_for_timeout(10000)
    response = page.inner_text("div[data-message-author-role='assistant']")

    browser.close()
    return response.strip()

def evaluar_gemini(playwright, texto, prompt):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(storage_state="gemini_state.json")  # cookies guardadas
    page = context.new_page()
    page.goto("https://gemini.google.com/")

    page.wait_for_timeout(3000)
    page.fill('textarea', f"{prompt}\n\n{texto}")
    page.keyboard.press("Enter")

    page.wait_for_timeout(10000)
    response = page.inner_text("div.response")  # Ajustar selector con DevTools

    browser.close()
    return response.strip()


# === Evaluación principal ===
with sync_playwright() as playwright:
    for p in prompts_folder:
        prompt_name = p.split("\\")[-1].replace(".txt", "")  # nombre del prompt
        with open(p, 'r', encoding='utf-8') as file:
            prompt_text = file.read()

        for resultado_item in resultados:
            texto = resultado_item["transcripcion"]

            # Crear sección para este prompt si no existe
            if prompt_name not in resultado_item["resultados"]:
                resultado_item["resultados"][prompt_name] = {}

            # Ejecutar cada modelo
            for modelo, funcion in {
                "ChatGPT": evaluar_chatgpt,
                "Gemini": evaluar_gemini
            }.items():
                try:
                    respuesta = funcion(playwright, texto, prompt_text)
                    calificacion = json.loads(respuesta)  # convertir a diccionario
                except Exception as e:
                    calificacion = {"error": str(e)}

                resultado_item["resultados"][prompt_name][modelo] = calificacion

# === Guardar resultados ===
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("✅ Resultados guardados en 'resultados.json'")
