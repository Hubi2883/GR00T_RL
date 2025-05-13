import pandas as pd
import pyarrow.parquet as pq

# show every column when printing a DataFrame
pd.set_option('display.max_columns', None)
# optionally also expand the width so lines won't wrap
pd.set_option('display.width', None)

# Read only the specified columns
columns = ["annotation.human.validity", "next.reward"]
table = pq.read_table(
    "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_results_0001/data/chunk-000/episode_000010.parquet"#, 
    #columns=columns
)
df = table.to_pandas()

# Display the filtered data
print(f"Showing only {len(columns)} columns: {', '.join(columns)}")
print(df)