from argparse import ArgumentParser

from data.utils import train_test_split

def parse_paths():
    """ TODO """
    parser = ArgumentParser()
    parser.add_argument('--datadir',
                        dest='data_dir',
                        type=str,
                        help='path to the rainy images directory')

    parser.add_argument('--gtruthdir',
                        dest='g_truth_dir',
                        type=str,
                        help='path to the clear images directory')
    parser.add_argument('--destdir',
                        dest='dest_dir',
                        type=str,
                        help="""path to the directory where the splited dataset
                        will be stored""")
    return parser.parse_args()


if __name__ == "__main__":
    paths = parse_paths()
    train_test_split(paths.data_dir,
                     paths.g_truth_dir,
                     paths.dest_dir,
                     ratio=0.9)
