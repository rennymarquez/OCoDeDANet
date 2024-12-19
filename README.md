# dynamic-cls

# DESCRIPTION
    dyncls [-h] --model {SNMF,SBNMF,SUBNMF,DBNMFARD,DUBNMFARD} [--initial-K INITIAL_K] [--n-iters N_ITERS] [--min-iters MIN_ITERS] [--a A] [--b B] [--tolerance TOLERANCE] [--alpha ALPHA]
              [--matrix-seed MATRIX_SEED] [--csv-file CSV_FILE]
              data
# OPTIONS
  positional arguments:
    data                  Data file path

  options:
    -h, --help            show this help message and exit
    --model {SNMF,SBNMF,SUBNMF,DBNMFARD,DUBNMFARD}
                          The model to apply to the data.
    --initial-K INITIAL_K
                          Number of initial communities
    --n-iters N_ITERS     Number of iterations
    --min-iters MIN_ITERS
                          Minimum number of iterations
    --a A                 Param a
    --b B                 Param b
    --tolerance TOLERANCE
                          Tolerance between matrices
    --alpha ALPHA         Param alpha
    --matrix-seed MATRIX_SEED
                          Matrix initialization seed
    --csv-file CSV_FILE   CSV file to store results
