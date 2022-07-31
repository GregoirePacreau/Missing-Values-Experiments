from auxiliary_functions import *
import numpy
from tqdm import tqdm
import pickle

def run_experiment(sample_size, dim_size, effective_rank, n_exp, epsilons, output, contamination, method, intensity=0, seed=0, index=None):

    numpy.random.seed(seed)
    errors = {}

    assert isinstance(method, str) or isinstance(method, list), "methods should be a string or a list of string, got {}".format(type(method))
    if isinstance(method, str):
        method_name = method
        errors[method] = {}
    else:
        method_name = '+'.join(method)
        for k in method:
            errors[k] = {}

    for epsilon in tqdm(epsilons):

        # initialising the experiments in dictionary
        for k in errors.keys():
            errors[k][epsilon] = []

        for _ in range(n_exp):

            #generate data
            sigma,_ = low_rank(effective_rank, dim_size, sample_size)
            X = numpy.random.multivariate_normal(numpy.zeros(dim_size), sigma, size=sample_size)
            # contaminate data
            if contamination == "bernoulli":
                X_noisy = contaminate_bernoulli(X, epsilon, intensity)
            elif contamination == "adversarial":
                X_noisy = contaminate_adversarial(X, sigma, epsilon)

            #compute error of each method
            for k in errors.keys():
                sigma_hat = apply_estimator(k, X_noisy)
                error = compute_error(sigma, sigma_hat, index=index)
                errors[k][epsilon].append(error)


    filename = 'exp_n={}_p={}_e={}_n_exp={}_cont={}_meth={}.pkl'.format(
        sample_size,
        dim_size,
        effective_rank,
        n_exp,
        contamination,
        method_name
    )
    with open(output+filename, 'w+') as file:
        pickle.dump(errors, file)