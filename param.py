import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MOGLAM.")

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of conv layers.')
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='Number of hidden units per modal.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of attention head.')
    parser.add_argument('--nmodal', type=int, default=3,
                        help='Number of omics.')

    return parser.parse_args()

