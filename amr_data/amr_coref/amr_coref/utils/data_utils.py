import gzip
import json


# Write json data in standard or gzipped format
def dump_json(data, fn):
    if fn.endswith('.gz'):
        with gzip.open(fn, 'w') as f:
            f.write(json.dumps(data, indent=4).encode('utf-8'))
    else:
        with open(fn, 'w') as f:
            json.dump(data, f, indent=4)

# Read json data in standard or gzipped format
def load_json(fn):
    if fn.endswith('.gz'):
        with gzip.open(fn, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))
    else:
        with open(fn) as f:
            data = json.load(f)
    return data
