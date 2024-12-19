from argparse import ArgumentParser
import pathlib

class Config():
    """Docstring for Config. """

    def __init__(self):
        """TODO: to be defined. """
        parser = ArgumentParser()
        parser.add_argument('--model', choices=['SNMF', 'SBNMF', 'SUBNMF', 'DBNMFARD', 'DUBNMFARD', 'GDUBNMFARD', 'DXBNMFARD', 'AADBNMFARD', 'CUSTOM'], \
                help='The model to apply to the data.', required=True)
        parser.add_argument('--initial-K', type=int, \
                help='Number of initial communities')
        parser.add_argument('--n-iters', type=int, default=5000, \
                help='Number of iterations')
        parser.add_argument('--min-iters', type=int, default=200, \
                help='Minimum number of iterations')
        parser.add_argument('--a', type=int, default=5, \
                help='Param a')
        parser.add_argument('--b', type=int, default=2, \
                help='Param b')
        parser.add_argument('--tolerance', type=float, default=1e-5, \
                help='Tolerance between matrices')
        parser.add_argument('--alpha', type=float, default=0.5, \
                help='Param alpha')
        parser.add_argument('--matrix-seed', type=int, default=1, \
                help='Matrix initialization seed')
        parser.add_argument('--csv-file', default=pathlib.Path('results.csv'), type=pathlib.Path, \
                help='CSV file to store results')
        parser.add_argument('data', type=pathlib.Path, \
                help='Data file path')
        args = parser.parse_args()

        self.model = args.model
        self.data = args.data
        self.n_iters = args.n_iters
        self.min_iters = args.min_iters
        self.csvfile = args.csv_file
        self._parser = parser
        self.alpha = args.alpha

    def model_params(self, model_config):
        """TODO: Docstring for model_params.
        :returns: TODO

        """

        self._parser.parse_args(namespace=model_config)
