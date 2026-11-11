"""Télécharge les poids des modèles de l'expérience, aux révisions exactes utilisées pour les résultats publiés.

À lancer sur un poste connecté à Internet ; copier ensuite le dossier dans l'environnement cible et le désigner par
LOCAL_MODELS_DIR : le notebook charge alors les modèles sans aucun accès réseau.

Usage :
    pip install huggingface_hub
    python download_models.py DOSSIER [--scale reduced|full]
"""
import argparse
from pathlib import Path

# Modèles chargés à chaque échelle (SCALE dans le notebook).
MODELS = {'reduced': ['almanach/camembert-base'], 'full': ['almanach/camembert-base']}
# Révision (commit Hugging Face) de chaque modèle : les mêmes poids que pour les résultats publiés.
REVISIONS = {'almanach/camembert-base': 'a75967561c78f2aa81cc41045378d3b4ee25af9e'}


def main() -> None:
    """Télécharge chaque modèle de l'échelle demandée dans DOSSIER/org--nom."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="dossier de destination")
    parser.add_argument("--scale", choices=sorted(MODELS), default="full", help="échelle (défaut : full)")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for model_id in MODELS[args.scale]:
        target = Path(args.folder) / model_id.replace("/", "--")
        snapshot_download(model_id, revision=REVISIONS.get(model_id), local_dir=target)
        print(f"{model_id} -> {target}")


if __name__ == "__main__":
    main()
