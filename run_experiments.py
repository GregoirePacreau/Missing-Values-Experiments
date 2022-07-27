from auxiliary_functions import *
import numpy
import tqdm
import pickle

def run_experiment(sample_size, dim_size, effective_rank, n_exp, epsilons, output, contamination, method, intensity=0, seed=0, index=None):

    numpy.random.seed(seed)
    errors = {}

    assert isinstance(method, str) or isinstance(method, list), "methods should be a string or a list of string, got {}".format(type(method))
    if isinstance(method, str):
        errors[method] = {}
    else:
        for k in method:
            errors[k] = {}

    for epsilon in tqdm(epsilons):

        # initialising the experiments in dictionary
        for k in errors.keys():
            errors[k][epsilon] = []

        for _ in range(n_exp):

            #generate data
            sigma,_ = low_rank(effective_rank, dim_size, sample_size)
            X = numpy.random.multivariate_normal(sample_size, dim_size, sigma)

            # contaminate data
            if contamination == "bernoulli":
                X_noisy = contaminate_bernoulli(X, epsilon, intensity)
            elif contamination == "adersarial":
                X_noisy = contaminate_adversarial(X, sigma, epsilon)

            #compute error of each method
            for k in errors.keys():
                sigma_hat = apply_estimator(k, X_noisy)
                error = compute_error(sigma, sigma_hat, index=index)
                errors[k][epsilon].append(error)

    