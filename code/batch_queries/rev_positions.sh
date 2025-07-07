#!/bin/bash
#$ -cwd
#$ -l m_mem_free=24G
echo "starting job at `date`"
source ~/virtualenv/bin/activate
python3 rev_positions.py &> rev_positions.out
echo "ending job at `date`"


