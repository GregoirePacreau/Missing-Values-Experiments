import argparse
import numpy
from run_experiments import run_experiment

parser=argparse.ArgumentParser(
    description="main experiment framework"
)
parser.add_argument(
    '--dim_size',
    type=int,
    default=100
)
parser.add_argument(
    '--sample_size',
    type=int,
    default=1000
)
parser.add_argument(
    '--effective_rank',
    type=int,
    default=5
)
parser.add_argument(
    '--method',
    type=str,
    choices=["DI",
             "TSGS",
             "random_MV",
             "DDC_MV",
             "tail_MV",
             "all"],
    default="TSGS"
)
parser.add_argument(
    '--contamination',
    type=str,
    choices=["adversarial",
             "bernoulli"],
    default="bernoulli"
)
parser.add_argument(
    '--min_epsilon',
    type=float,
    default=0.01
)
parser.add_argument(
    '--max_epsilon',
    type=float,
    default=0.20
)
parser.add_argument(
    '--n_epsilon',
    type=int,
    default=30
)
parser.add_argument(
    '--n_exp',
    type=int,
    default=100
)
parser.add_argument(
    '--output',
    type=str,
    default="results/"
)

args = parser.parse_args()

if __name__ == '__main__':
    if args.method == "all":
        methods = ["DI", "TSGS", "random_MV", "DDC_MV", "tail_MV"]
    else:
        methods = args.method

    epsilons = numpy.linspace(args.min_epsilon, args.max_epsilon, args.n_epsilon)

    run_experiment(args.sample_size,
                    args.dim_size,
                    args.effective_rank,
                    args.n_exp,
                    epsilons,
                    args.output,
                    args.contamination,
                    methods)