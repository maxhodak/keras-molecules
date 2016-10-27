import argparse
import gzip
import pandas
import tempfile
import urllib

DEFAULT_URI = 'ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_22_chemreps.txt.gz'

def get_arguments():
    parser = argparse.ArgumentParser(description='Download ChEMBL entries and convert them to input for preprocessing')
    parser.add_argument('--uri', type=str, default=DEFAULT_URI,
                        help = 'URI to download ChEMBL entries from')
    parser.add_argument('--outfile', type=str, help='Output file name', default = 'chembl_smiles.h5')
    return parser.parse_args()

def main():
    args = get_arguments()
    tfile = tempfile.NamedTemporaryFile()
    fname = tfile.name

    urllib.urlretrieve(args.uri, fname)

    f = gzip.GzipFile(fname)
    d = {}
    for line in f.readlines()[1:]:
        s = line.split()
        i = int(s[0][6:])
        d[i] = s[1]

    keys = d.keys()
    keys.sort()
    frame = pandas.DataFrame(dict(structure=[d[key] for key in keys]))
    frame.to_hdf(args.outfile, 'table')

if __name__ == '__main__':
    main()
