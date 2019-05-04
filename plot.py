#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

paths = glob("*.txt")
plt.figure()
plt.xlabel('Number of Iterations')
plt.ylabel('Objective function value')
plt.title('Archetype optimization')
for path in paths:
	print(path)
	with open(path) as f:
	    lines = f.readlines()
	    values = []
	    for line in lines:
	    	value = float(line)
	    	values.append(value)
	    layer = path.split('.')[0]
	    print(layer)
	    plt.plot(range(len(values)), values, label=layer)
	    plt.legend()
plt.savefig('plot.png')
plt.close()