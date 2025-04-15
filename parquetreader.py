import pyarrow.parquet as pq




# Read the Parquet file into a PyArrow Table
table = pq.read_table("/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_results_0002action/data/chunk-000/episode_000002.parquet")
# Convert the table to a pandas DataFrame if needed
df = table.to_pandas()
print(df)
