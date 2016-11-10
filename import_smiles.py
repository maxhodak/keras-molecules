import gzip
import pandas
def read_smiles(fname, column=0):
    if fname.endswith("gz"):
        f = gzip.GzipFile(fname)
    else:
        f = open(fname)
    
    d = {}
    i = 0
    for line in f.readlines()[1:]:
        s = line.split()
        d[i] = s[column]
        i = i + 1

    return d

def create_h5(d, outfile):
    keys = d.keys()
    keys.sort()
    frame = pandas.DataFrame(dict(structure=[d[key] for key in keys]))
    frame.to_hdf(outfile, 'table')
