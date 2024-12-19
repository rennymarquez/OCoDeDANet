from .config import Config
from .data import Data
from .solver import Solver

def main() -> None:
    config = Config()
    data = Data(config.data)
    solver = Solver(data, config)
    solver.solve()
