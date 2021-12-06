import os
import numpy as np
import yaml


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def get_freer_gpus(n):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    ind = np.argsort(memory_available)
    return np.argmax(memory_available)[-n:]


def smooth(smoothed, scalars, weight):  # weight between 0 and 1
    assert len(scalars) - len(smoothed) == 1

    if len(scalars) == 1:
        smoothed.append(scalars[0])
    else:
        smoothed.append(weight * smoothed[-1] + (1 - weight) * scalars[-1])

    return smoothed


def moving_average(scalars, weight):
    smoothed = []
    for i in range(len(scalars)):
        smooth(smoothed, scalars[: i + 1], weight)
    return smoothed


def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    for key, value in runcard.items():
        if ("hp." in str(value)) or ("None" == str(value)):
            runcard[key] = eval(value)
    return runcard
