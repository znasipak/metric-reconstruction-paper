import os
import json
import pandas as pd

def build_metadata(data_dir):
    rows = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        # find any .json files in this directory
        json_files = [f for f in filenames if f.lower().endswith('.json')]
        if not json_files:
            continue
        # subdir relative to data_dir
        subdir = os.path.relpath(dirpath, data_dir)
        for jf in json_files:
            params_path = os.path.join(dirpath, jf)
            try:
                with open(params_path, 'r') as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"Warning: failed to read {params_path}: {e}")
                data = {}

            # extract expected fields if present
            gauge = data.get('gauge', '')
            name = data.get('jobname', '')
            a = data.get('a', '')
            p = data.get('p', '')
            e = data.get('e', '')
            x = data.get('x', '')
            lmax = data.get('lmax', '')

            base, _ = os.path.splitext(jf)
            huu_name = base.replace('params', 'huuYl') + '.npy'

            rows.append({
                'subdir': subdir,
                'gauge': gauge,
                'name': name,
                'a': a,
                'p': p,
                'e': e,
                'x': x,
                'lmax': lmax,
                'params_file': jf,
                'huu_file': huu_name,
            })

    df = pd.DataFrame(rows, columns=['subdir','gauge','name','a','p','e','x','lmax','params_file','huu_file'])
    return df


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "huu")
    data_dir = os.path.normpath(data_dir)
    print(f"Scanning data directory: {data_dir}")
    df = build_metadata(data_dir)
    # sort rows by first three columns: subdir, gauge, name
    if not df.empty:
        df.sort_values(by=['subdir', 'name'], inplace=True, na_position='last')
        df.reset_index(drop=True, inplace=True)
    out_csv = os.path.join(data_dir, 'huu_metadata.csv')
    df.to_csv(out_csv, index=False)
    print(f"Wrote metadata for {len(df)} entries to {out_csv}")
    
