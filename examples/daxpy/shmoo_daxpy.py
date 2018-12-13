#!/usr/bin/env python3

import sys
import os
import subprocess
import numpy

endpoints = numpy.logspace(3, 8.66, 10)

sizes = []
start = 1
for end in endpoints:
  points = numpy.linspace(start, end, 10, endpoint = False)
  sizes.extend([int(pt) for pt in points])
  start = end
sizes.append(int(start))

program_name = os.path.abspath(sys.argv[1])

print("size,bandwidth")

for size in sizes:
    output = subprocess.run([program_name, str(size)], stdout = subprocess.PIPE, stderr = subprocess.PIPE).stderr
    
    print(output.decode(), end = '')

