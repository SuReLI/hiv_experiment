#!/usr/bin/env python
import matplotlib.pyplot as plt

from fqi import fqi


if __name__ == "__main__":
    plt.style.use("dark_background")

    params = {
        'n': 100,
        't': 1000,
        "eval": 10,
        "eval_t": 750,
        "gamma": 0.98,
        'Q': 0.1,
        "R1": 20000,
        "R2": 2000,
        'S': 1000
        }
    rf_star = fqi(**params)
