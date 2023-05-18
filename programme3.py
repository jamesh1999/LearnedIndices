import subprocess

DATASETS = [
    ("fashionmnist", [100, 20]),
    ("sifttrunc", [48, 12])
    ]
MODELS = ["basic", "triplet", "relaxed", "relaxed2", "vae"]
CONSTANT_FLAGS = ["--triplet_reps"]
CONSTANTS = [
    [50],
]
TYPES = [
    ["-t", "scae"],
]

for t in TYPES:
    for dataset, dims in DATASETS:
        for d in dims:
            for reps in [50]:
                subprocess.run(
                    map(str, [
                        "python", "train_scae.py", "-D", dataset, "-m", "basicnn", "-d", d, "--triplet_reps", reps
                    ] + t))