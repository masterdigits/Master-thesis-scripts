from genetic_abstract import elite_selection_model, genetic_algorithm
from GPU_engine import GPU_engine
from Variant import Variant
from variant import variant
from Chooser import Chooser
from Chromosom import Chromosom

import cupy as cp
import cudf as df
from datetime import datetime
from random import choice,random
from copy import copy
import pandas as pd
import numpy as np

import plotly.express as px
from collections import Counter

def funkcja(pop1,pop2):
    chromosom = q_in.get()
    f1 = f'/home/dkrawczyk/lefik/gpu_space/synthetic_population1.csv.gz'
    f2 = f'/home/dkrawczyk/lefik/gpu_space/synthetic_population2.csv.gz'
    files = (f1, f2)
    chr_obj = Chromosom(
        name=chromosom,
        tables=files,
        maf_goodness=(.05,.95),
        chooser_groups_number=20000,
        chooser_length=1000,
        E=500,
        E_learn=50,
        N=5000,
        variants_number=150
    )
    chr_obj.run()
    q_out.put(chr_obj)
    return

def thread(pop1,pop2):
    while ~q_in.empty():
        funkcja(pop1,pop2)
    return

if __name__ == "__main__":
    #threads = []
    q_in = Queue()
    for i in range(1):
        q_in.put(f'chr{i}')
    q_out = Queue()
    #for k in range(10):
    #    t = Thread(target=thread, args=('FIN','YRI'))
    #    threads.append(t)
    #    t.start()
    thread('1','1')