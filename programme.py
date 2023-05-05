import subprocess

DATASETS = ["fashionmnist"]
MODELS = ["universal"]
DIMENSIONS = [40]
CONSTANT_FLAGS = ["--crecon","--calpha","--clambda","--cortho","--cspace","--ctriplet","--triplet_reps", "--encoder_wd", "--decoder_wd"]
CONSTANTS = [
    [4,1,0,0,0,0.25,50,0,0],
    [4,1,0,0,0,0.5,50,0,0],
    [4,1,0,0,0,1,50,0,0],
    [4,1,0,0,0,2,50,0,0],
    [4,1,0,0,0,4,50,0,0],
]

for dataset in DATASETS:
    for model in MODELS:
        for dims in DIMENSIONS:
            for cs in CONSTANTS:
                constants = []
                for i, f in enumerate(CONSTANT_FLAGS):
                    constants += [f, cs[i]]
                subprocess.run(map(str, ["python", "train_models.py", "-D", dataset, "-m", "basicnn", "-t", model, "-d", dims] + constants))
