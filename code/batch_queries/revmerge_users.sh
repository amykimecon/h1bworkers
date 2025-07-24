#!/bin/bash
#$ -cwd
#$ -l m_mem_free=24G
echo "starting job at `date`"
source ~/virtualenv/bin/activate
python3 revmerge_users.py &> revmerge_users.out
python3 chunk_merge.py &>> revmerge_users.out
echo "ending job at `date`"


