import pandas as pd

# show every column when printing a DataFrame
pd.set_option('display.max_columns', None)
# optionally also expand the width so lines won’t wrap
pd.set_option('display.width', None)

# now load and show
import pyarrow.parquet as pq
table = pq.read_table("/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/Data_results_0001/Data/chunk-000/episode_000001.parquet")
df = table.to_pandas()
print(df)
