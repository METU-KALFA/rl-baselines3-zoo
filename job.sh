#!/bin/sh
python train.py --algo ppo --env assembly-cloud-partial-chamfer-v0 -tb assembly_cloud --verbose 0 --w-group chamfer
# python train.py --algo ppo --env assembly-cloud-end-chamfer-v0 -tb assembly_cloud --verbose 0 --w-group chamfer
