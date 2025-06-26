import subprocess
import pandas as pd
from io import StringIO
import os 

mdb_root = '/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers/data/raw/lca/'

for file in os.listdir(mdb_root):
    if file.endswith(".mdb"):
        tables = subprocess.check_output(['mdb-tables', '-1', f"{mdb_root}{file}"]).decode().splitlines()
        
        for table in tables:
            output = subprocess.check_output(['mdb-export', f"{mdb_root}{file}", table])
            df = pd.read_csv(StringIO(output.decode()), sep=',')
            df.to_csv(f"{mdb_root}{table}.csv", index = False)
