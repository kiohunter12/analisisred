# main.py
import os

print("🚀 INICIANDO PROYECTO DE RED NEURONAL DE POBREZA...\n")

os.system("python scripts/preprocesamiento.py")
os.system("python scripts/entrenamiento.py")
os.system("python scripts/evaluacion.py")
os.system("python scripts/mapa_calor.py")

print("\n✅ PROCESO COMPLETO FINALIZADO.")
