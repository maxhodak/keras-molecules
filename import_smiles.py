import argparse
import gzip
import pandas

DEFAULT_SMILES="test.smi"

def get_arguments():
    parser = argparse.ArgumentParser(description='Download ChEMBL entries and convert them to input for preprocessing')
    parser.add_argument('smiles', type=str, default=DEFAULT_SMILES,
                        help = 'URI to download ChEMBL entries from')
    parser.add_argument('--outfile', type=str, help='Output file name', default = 'smiles.h5')
    parser.add_argument('--column', type=int, help='Input column', default = 0)
    return parser.parse_args()

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

def main():
    args = get_arguments()
    d = read_smiles(args.smiles, column=args.column)
    create_h5(d, args.outfile)

if __name__ == '__main__':
    main()
