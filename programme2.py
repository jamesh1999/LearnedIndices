import subprocess

MODELS = [7, 9, 11, 19, 21, 23]

INDEX_TYPES = [
        ("kdtree", -1),
        ("pynn", 5), ("pynn", 10), ("pynn", 25), ("pynn", 50),
        ("hnsw", 5), ("hnsw", 10), ("hnsw", 25), ("hnsw", 50)
    ]

for model in MODELS:
    for index in INDEX_TYPES:
        subprocess.run(map(str,["python", "build_indices.py", "-m", model, "-I", index[0], "-i", index[1]]))