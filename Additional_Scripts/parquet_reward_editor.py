import os
import argparse
import pandas as pd


def process_file(file_path: str, interval: float = 2.0, transition: int = 50):
    """
    Reads a parquet file, prompts user at approximately every `interval` seconds based on the timestamp column
    to set annotation.human.validity, applies that value across the chunk, and updates next.reward with a delayed effect.

    Parameters:
        file_path: Path to the parquet file.
        interval: Time interval in seconds for prompting.
        transition: Number of rows for reward transition when validity changes.
    """
    # Load data
    df = pd.read_parquet(file_path)
    required_cols = {'timestamp', 'annotation.human.validity', 'next.reward'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing columns {missing} in {file_path}")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    start_time = df['timestamp'].iloc[0]
    max_time = df['timestamp'].iloc[-1]

    prev_idx = 0
    prev_time = start_time
    prev_validity = int(df['annotation.human.validity'].iloc[0]) if not df['annotation.human.validity'].isna().all() else 0

    # Iterate by time intervals
    while True:
        target_time = prev_time + interval
        if target_time > max_time:
            break

        # Find closest row to target_time
        idx_end = (df['timestamp'] - target_time).abs().idxmin()
        chunk_start = prev_idx
        chunk_end = idx_end + 1

        # Prompt for validity
        while True:
            resp = input(
                f"Timestamps {df['timestamp'].iloc[chunk_start]:.2f}-"
                f"{df['timestamp'].iloc[idx_end]:.2f}: set annotation.human.validity (0 or 1): "
            ).strip()
            if resp in {'0', '1'}:
                validity = int(resp)
                break
            print("Invalid input. Enter 0 or 1.")

        # Apply validity
        df.loc[chunk_start:chunk_end - 1, 'annotation.human.validity'] = validity

        # Update next.reward with delayed effect
        for offset in range(chunk_end - chunk_start):
            idx = chunk_start + offset
            if validity == prev_validity:
                reward = validity
            else:
                reward = prev_validity if offset < transition else validity
            df.at[idx, 'next.reward'] = reward

        prev_idx = chunk_end
        prev_time = df['timestamp'].iloc[idx_end]
        prev_validity = validity

    # Tail rows
    if prev_idx < len(df):
        df.loc[prev_idx:, 'annotation.human.validity'] = prev_validity
        df.loc[prev_idx:, 'next.reward'] = prev_validity

    # Save back
    df.to_parquet(file_path, index=False)
    print(f"Updated: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process all parquet files under the fixed DATA_DIR by time intervals."
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Seconds between prompts (default: 2.0)."
    )
    parser.add_argument(
        "--transition", type=int, default=50,
        help="Rows for reward transition (default: 50)."
    )
    args = parser.parse_args()

    # Fixed root directory
    DATA_DIR = r"C:\Users\trege\OneDrive\Desktop\P8\RT1\Data_results_0001\Data\chunk-000"

    for dirpath, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.endswith('.parquet'):
                file_path = os.path.join(dirpath, fname)
                try:
                    print(f"Processing {file_path}...")
                    process_file(file_path, args.interval, args.transition)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


if __name__ == '__main__':
    main()