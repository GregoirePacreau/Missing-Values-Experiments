from source.auxiliary_functions import *
import numpy
from tqdm import tqdm
import pickle

def run_experiment(sample_size,
                    dim_size,
                    effective_rank,
                    n_exp, epsilons,
                    output,
                    contamination,
                    method,
                    cont_option=None,
                    intensity=None, 
                    seed=0,
                    index=None,
                    return_time=True):

    filename = 'nsample={}_dim={}_erank={}_nexp={}_cont={}_meth={}_intensity={}_cont-type={}.pkl'.format(
        sample_size,
        dim_size,
        effective_rank,
        n_exp,
        contamination,
        method,
        intensity,
        cont_option
    )

    if not os.path.exists(output+filename):

        numpy.random.seed(seed)
        estimates = {}
        exec_times = {}

        estimates["truth"] = {}

        assert isinstance(method, str) or isinstance(method, list), "methods should be a string or a list of string, got {}".format(type(method))
        if isinstance(method, str):
            method_name = method
            estimates[method] = {}
            exec_times[method] = {}
        else:
            method_name = '+'.join(method)
            for k in method:
                estimates[k] = {}
                exec_times[k] = {}

        methods = list(estimates.keys())
        methods.remove("truth")

        pbar = tqdm(epsilons)
        pbar.set_description(method + ' ' + contamination)
        for epsilon in pbar:

            # initialising the experiments in dictionary
            for k in estimates.keys():
                estimates[k][epsilon] = []
            for k in methods:
                exec_times[k][epsilon] = []

            for _ in range(n_exp):

                #generate data and saving the true covariance matrix
                sigma,_ = low_rank(effective_rank, dim_size)
                estimates["truth"][epsilon].append(sigma)

                X = numpy.random.multivariate_normal(numpy.zeros(dim_size), sigma, size=sample_size)
            
                # contaminate data
                if contamination == "bernoulli":
                    X_noisy, mask = contaminate_bernoulli(X, epsilon, intensity, option=cont_option)
                elif contamination == "adversarial":
                    X_noisy = contaminate_adversarial(X, sigma, epsilon)
                    mask = np.zeros(X.shape)
                #compute and save covaraince estimates
                for k in methods:
                    sigma_hat, exec_time = apply_estimator(k, X_noisy, mask)
                    
                    estimates[k][epsilon].append(sigma_hat)
                    exec_times[k][epsilon].append(exec_time)

        with open(output+filename, 'wb+') as file:
            pickle.dump(estimates, file)
        if return_time:
            with open(output+'exectime_'+filename, 'wb+') as file:
                pickle.dump(exec_times, file)
    else:
        print("{} already exists".format(filename))