import os

# Racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin global vers mlruns
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")