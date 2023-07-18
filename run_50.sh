#!/bin/bash

module load prun

for i in {1..13}
do
  prun -np 1 python pointnet2_classification.py
done