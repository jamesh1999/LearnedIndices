import subprocess

DATASETS = [
    ("nytrunc", [128, 26]),
    ("lasttrunc", [48, 20])
    ]
MODELS = ["basic", "triplet", "relaxed", "relaxed2", "vae"]
CONSTANT_FLAGS = ["--triplet_reps"]
CONSTANTS = [
    # [1],
    # [2],
    # [4],
    # [8],
    # [16],
    # [28],
    [50],
]
TYPES = [
    ["-t", "basic"],

    ["-t", "relaxedold"],

    ["-t", "relaxed"],

    # ["-t", "vae", "--calpha", 1, "--clambda", 0], # AE
    # ["-t", "vae", "--calpha", 0, "--clambda", 1], # VAE

    ["-t", "triplet"],
]

for t in TYPES:
    for dataset, dims in DATASETS:
        for d in dims:
            for reps in [50]:
                subprocess.run(
                    map(str, [
                        "python", "train_models.py", "-D", dataset, "-m", "basicnn", "-d", d, "--triplet_reps", reps
                    ] + t))