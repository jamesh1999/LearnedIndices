import subprocess

DATASETS = ["fashionmnist"]
MODELS = ["vae"]
DIMENSIONS = [800, 400, 200, 100, 50, 25]

for dataset in DATASETS:
    for model in MODELS:
        for dims in DIMENSIONS:
            subprocess.run(map(str, ["python", "train_models.py", "-D", dataset, "-m", "basicnn", "-t", model, "-d", dims]))