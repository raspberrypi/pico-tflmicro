#!/usr/bin/env python

import sys

REPLACEMENT_PATTERN = sys.argv[3]

with open(sys.argv[2]) as f:
    replacement_contents = f.read()

with open(sys.argv[1], "rt") as fin:
    for line in fin:
        sys.stdout.write(line.replace(
            REPLACEMENT_PATTERN, replacement_contents))
