# main auxiliary functions for missing values experiments

import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.covariance import MinCovDet

##### Low rank matrix generation #####

def low_rank(effective_rank, p, n, bell_curve=False):
    """
    Generates a low rank matrix using one of two spectral profile

    Parameters
    ----------
    effective_rank : TYPE
        Effective rank of the matrix.
    p : TYPE
        Dimension of the square matrix to be produced.
    bell_curve : TYPE, optional
        Whether to substitute the exponentially declining eigenvalues with
        a bell curve like eigenvalue structure. The default is False.

    Returns
    -------
    a tuple composed of the generated covariance matrix and the tuple of its eigenvalue decomposition.

    """
    
    ## Index of the singular values 
    singular_ind = numpy.arange(p, dtype=numpy.float64)
    
    if bell_curve:
        
        # Build the singular profile by assembling signal and noise components (from the make_low_rank_matrix code of numpy)
        tail_strength = 0.5
        low_rank = (1 - tail_strength) * numpy.exp(-1.0 * (singular_ind / effective_rank) ** 2)
        tail = tail_strength * numpy.exp(-0.1 * singular_ind / effective_rank)
        eigenvalues = (low_rank + tail)
    
    else:
        # Build the singular profile using an exponential:
        eigenvalues = effective_rank/2*numpy.exp(-2*singular_ind/(effective_rank))

    diag= numpy.eye(p)*eigenvalues
    
    # Generating random orthonormal vectors
    H = numpy.random.randn(p, p)
    u, s, vh = numpy.linalg.svd(H)
    mat = u @ vh

    sigma = mat @ diag @ mat.T

    return sigma, (mat, eigenvalues)

def estimate_cov(X, delta, mask, bias=True):
    """
    Unbiased covariance estimation under missing from Lounici (2014)

    Parameters
    ----------
    X : (n,p) numpy array or similar
        The samples from which to estimate the covariance. The covariance matrix will have shape (p,p).
    delta : float
        probability of the bernoulli random variable governing whether we see the values or not.
    mask : (n,p) boolean array or similar
        The boolean mask hiding values from the estimator, optional if X already has missing values.
    bias : bool, optional
        Whether to use the unbiased version of the classical estimator, should not have much impact. The default is True.

    Returns
    -------
    sigma_tilde : (p,p) numpy array
        Estimated covariance matrix.

    """
    Y = mask * X
    sigma = numpy.cov(Y.T, bias=bias)
    sigma_tilde = (1/delta - 1/(delta**2)) * numpy.eye(sigma.shape[0]) * sigma.diagonal() + 1/(delta**2) * sigma
    return sigma_tilde

##### Adversarial perturbations #####

def contaminate_bernoulli(X, epsilon, intensity=1):
    """
    bernoulli contamination of a gaussian random variable: random elements are
    replaced by a uniform law in [-1,1] 
    """
    bernoulli_mask = numpy.random.binomial(p=epsilon, n=1, size=X.shape)
    #noise = intensity*2*numpy.random.randn(*X.shape) - intensity
    noise = intensity*numpy.ones(shape=X.shape)
    X_contaminated = X*(1-bernoulli_mask) + noise*bernoulli_mask
    return X_contaminated

def random_sparse_sphere(p, s):
    # generates s sparse vectors on the unit sphere
    sparsity_padding = numpy.array([0]*(p-s))
    coord = numpy.random.normal(0, 1, size=s)
    coord = numpy.concatenate([sparsity_padding, coord])
    coord = numpy.random.permutation(coord)
    norm = numpy.linalg.norm(coord)
    return coord/norm

def find_sparse_orthogonal(u, s=5, max_iter=1000):
    # Finds the sparsest unitary vector orthogonal to u

    # actually approximate experiment, where we generate s sparse vectors and select the one of lowest scalar product with u
    
    min_inner_product = 10
    best_vector = None

    for k in range(int(max_iter)):
        v = random_sparse_sphere(u.shape[0], s)
        inner_product = numpy.abs(v@u)
        if best_vector is None or inner_product < min_inner_product:
            min_inner_product = inner_product
            best_vector = v
        if inner_product == 0:
            break
    return best_vector

def contaminate_adversarial(X, sigma, epsilon=0.05, max_iter=1e3):
    """
    Adversarial contamination of X based on sparse vectors orthogonal to the first eigenspace of sigma.

    Parameters
    ----------
    X : (n,p) array of similar
        Samples of a mutlivariate distribution with covariance sigma.
    sigma : (p,p) array or similar
        Covariance matrix of X.
    s : sparsity parameter of the adversarial vector, ie number of non zero components (default is 5)
    max_iter : number of iterations in the search for the most orthogonal sparse vector.

    Returns
    -------
    X_noisy : (n,p) numpy array
        Contaminated samples of X.

    """
    u,d,v = numpy.linalg.svd(sigma)
    s = int(len(v[0]) * epsilon)
    theta_adv = find_sparse_orthogonal(v[0], s=s, max_iter=1e5)
    sigma_norm = numpy.linalg.norm(sigma)

    contaminated_row_mask = numpy.random.binomial(n=1, p=0.05, size=X.shape[0])
    gaussian_noise = numpy.random.normal(0, 1, X.shape)
    contamination = numpy.outer(contaminated_row_mask, theta_adv)

    X_noisy = X + numpy.sqrt(2)*sigma_norm*gaussian_noise*contamination
    return X_noisy

##### Robust estimators #####

def MV_estimator(X, delta=0.9, N=1):
    """
    Estimator based on the random hiding of values followed by the unbiased estimator of Lounici (2014)

    Parameters
    ----------
    X : (n,p) array or similar.
        The contaminated data.
    delta : float between 0 and 1, optional
        Probability parameter of the bernoulli law for the random mask. The default is 0.9.
    N : int, optional
        Number of random bernoulli masks to be used during the estiamtion. The final matrix will be averaged
        over all the masks. The default is 1.

    Returns
    -------
    A (p,p) estimated covariance matrix.

    """
    mask = numpy.random.binomial(size=X.shape, n=1, p=delta)
    estimated_covs = estimate_cov(X, delta, mask)
    for k in range(N-1):
        mask = numpy.random.binomial(size=X.shape, n=1, p=delta)
        estimated_covs += estimate_cov(X, delta, mask)
    return estimated_covs/N

def find_perturbation(X, threshold_ratio):
    # under assumption that X follows a multivariate normal distribution
    stds = numpy.std(X, axis=0)
    stds_good_shape = numpy.outer(numpy.ones(X.shape[0]), stds)
    mask_hat = X <= ((threshold_ratio) * stds_good_shape)
    mask_hat = numpy.multiply(mask_hat, 1) # bool -> int
    delta_hat = numpy.count_nonzero(mask_hat)/mask_hat.shape[0]/mask_hat.shape[1]
    return(mask_hat, delta_hat)

def MV_thresholding_estimator(X, threshold_ratio):
    mask_thresh, delta_thresh = find_perturbation(X, threshold_ratio)
    return estimate_cov(X, delta_thresh, mask_thresh)

##### Comparison experiments #####

def get_proj(sigma, index):
    v = numpy.linalg.svd(sigma)[2][index]
    return numpy.outer(v, v.T)

def bernoulli_exp(n=500, p=100, epsilons=numpy.linspace(0.01, 0.2, 10), 
                    delta=0.9, M=100, e_rank=5, threshold_ratio=2, 
                    add_robust=False, eigen_index=None):
    # List of all the models used during this experiment
    error_classical = []
    std_classical = []
    error_robust = []
    std_robust = []
    error_thresh = []
    std_thresh = []
    error_MV = []
    std_MV = []

    for epsilon in tqdm(epsilons):

        classical = []
        robust = []
        MV = []
        thresh = []

        for k in range(M):
            # case where contamination is set at a constant value that is larger than a threshold
            sigma, eigen = low_rank(e_rank, p, n)
            X = numpy.random.multivariate_normal(n,p,sigma)

            # Finding sparse perturbation along orthogonal vector to first eigenvector
            u,s,v = numpy.linalg.svd(sigma)
            if eigen_index is not None:
                true_proj = numpy.outer(v[eigen_index], v[eigen_index].T)

            X_noisy = contaminate_bernoulli(X, epsilon)

            # Computing covariance matrices
            sigma_hat = MV_estimator(X_noisy, delta=delta)
            sigma_thresh = MV_thresholding_estimator(X_noisy)
            # robust MinCovDet estimator of sklearn, with high support fraction
            if add_robust:
                sigma_robust = MinCovDet(support_fraction=0.9, assume_centered=True).fit(X_noisy).covariance_
     
            if eigen_index is None:
                error_of_classic_cov = numpy.linalg.norm(sigma - numpy.cov(X_noisy.T, bias=True))
                if add_robust:
                    error_of_robust_cov = numpy.linalg.norm(sigma - sigma_robust)
                error_to_truth = numpy.linalg.norm(sigma - sigma_hat)
                error_thresh_to_truth = numpy.linalg.norm(sigma - sigma_thresh)
            else:
                proj_thresh = get_proj(sigma_thresh, eigen_index)
                proj_hat = get_proj(sigma_hat, eigen_index)

                #computing errors
                error_of_classic_cov = numpy.linalg.norm(true_proj - get_proj(numpy.cov(X_noisy.T, bias=True), eigen_index))
                if add_robust:
                    proj_robust = get_proj(sigma_robust, eigen_index)
                    error_of_robust_cov = numpy.linalg.norm(true_proj - proj_robust)
                error_to_truth = numpy.linalg.norm(true_proj - proj_hat)
                error_thresh_to_truth = numpy.linalg.norm(true_proj - proj_thresh)

            #appending errors to arrays
            classical.append(error_of_classic_cov)
            if add_robust:
                robust.append(error_of_robust_cov)
            MV.append(error_to_truth)
            thresh.append(error_thresh_to_truth)
      
        #computing means and std of each arrays
        error_classical.append(numpy.mean(classical))
        std_classical.append(numpy.std(classical))
        if add_robust:
            error_robust.append(numpy.mean(robust))
            std_robust.append(numpy.std(robust))
        error_MV.append(numpy.mean(MV))
        std_MV.append(numpy.std(MV))
        error_thresh.append(numpy.mean(thresh))
        std_thresh.append(numpy.std(thresh))

    # formatting for plots
    error_classical = numpy.array(error_classical)
    std_classical = numpy.array(std_classical)
    if add_robust:
        error_robust = numpy.array(error_robust)
        std_robust = numpy.array(std_robust)
    error_MV = numpy.array(error_MV)
    std_MV = numpy.array(std_MV)
    error_thresh = numpy.array(error_thresh)
    std_thresh = numpy.std(std_thresh)

    #plots
    fig, axs = plt.subplots(figsize=(10, 8))
    axs.plot(epsilons, error_classical, color='r')
    axs.fill_between(epsilons, (error_classical-std_classical), (error_classical+std_classical), color='r', alpha=.1)
    if add_robust:
        axs.plot(epsilons, error_robust, color='g')
        axs.fill_between(epsilons, (error_robust-std_robust), (error_robust+std_robust), color='g', alpha=.1)
    axs.plot(epsilons, error_MV, color='b')
    axs.fill_between(epsilons, (error_MV-std_MV), (error_MV+std_MV), color='b', alpha=.1)
    axs.plot(epsilons, error_thresh, 'c')
    axs.fill_between(epsilons, (error_thresh-std_thresh), (error_thresh+std_thresh), color='c', alpha=.1)
    axs.set_xlabel("Probability that a cell is contaminated")
    axs.set_ylabel("Frobenius error")
    if add_robust:
        axs.legend(["classical estimator", "Robust estimator", "MV estimator", "MV estimator with thresholding"])
    else:
        axs.legend(["classical estimator", "MV estimator", "MV estimator with thresholding"])
    if eigen_index is not None:
        axs.set_title("Frobenius error on {}th projector".format(eigen_index))
    else:
        axs.set_title("Forbenius error on covariance matrix estimations")
    plt.show()

def adv_exp(n=500, p=100, epsilons=numpy.linspace(0.01, 0.2, 10), delta=0.9, M=100, e_rank=5, threshold_ratio=2, add_robust=False, eigen_index=None):
    # List of all the models used during this experiment
    error_classical = []
    std_classical = []
    error_robust = []
    std_robust = []
    error_thresh = []
    std_thresh = []
    error_MV = []
    std_MV = []

    for epsilon in tqdm(epsilons):

        classical = []
        robust = []
        MV = []
        thresh = []

        for k in range(M):
            # case where contamination is set at a constant value that is larger than a threshold
            sigma, eigen = low_rank(e_rank, p, n)
            X = numpy.random.multivariate_normal(n,p,sigma)

            # Finding sparse perturbation along orthogonal vector to first eigenvector
            u,s,v = numpy.linalg.svd(sigma)
            if eigen_index is not None:
                true_proj = numpy.outer(v[eigen_index], v[eigen_index].T)

            X_noisy = contaminate_adversarial(X, sigma, epsilon=epsilon)

            # Computing covariance matrices
            sigma_hat = MV_estimator(X_noisy, delta=delta)
            sigma_thresh = MV_thresholding_estimator(X_noisy)
            # robust MinCovDet estimator of sklearn, with high support fraction
            if add_robust:
                sigma_robust = MinCovDet(support_fraction=0.9, assume_centered=True).fit(X_noisy).covariance_
     
            if eigen_index is None:
                error_of_classic_cov = numpy.linalg.norm(sigma - numpy.cov(X_noisy.T, bias=True))
                if add_robust:
                    error_of_robust_cov = numpy.linalg.norm(sigma - sigma_robust)
                error_to_truth = numpy.linalg.norm(sigma - sigma_hat)
                error_thresh_to_truth = numpy.linalg.norm(sigma - sigma_thresh)
            else:
                proj_thresh = get_proj(sigma_thresh, eigen_index)
                proj_hat = get_proj(sigma_hat, eigen_index)

                #computing errors
                error_of_classic_cov = numpy.linalg.norm(true_proj - get_proj(numpy.cov(X_noisy.T, bias=True), eigen_index))
                if add_robust:
                    proj_robust = get_proj(sigma_robust, eigen_index)
                    error_of_robust_cov = numpy.linalg.norm(true_proj - proj_robust)
                error_to_truth = numpy.linalg.norm(true_proj - proj_hat)
                error_thresh_to_truth = numpy.linalg.norm(true_proj - proj_thresh)

            #appending errors to arrays
            classical.append(error_of_classic_cov)
            if add_robust:
                robust.append(error_of_robust_cov)
            MV.append(error_to_truth)
            thresh.append(error_thresh_to_truth)
      
        #computing means and std of each arrays
        error_classical.append(numpy.mean(classical))
        std_classical.append(numpy.std(classical))
        if add_robust:
            error_robust.append(numpy.mean(robust))
            std_robust.append(numpy.std(robust))
        error_MV.append(numpy.mean(MV))
        std_MV.append(numpy.std(MV))
        error_thresh.append(numpy.mean(thresh))
        std_thresh.append(numpy.std(thresh))

    # formatting for plots
    error_classical = numpy.array(error_classical)
    std_classical = numpy.array(std_classical)
    if add_robust:
        error_robust = numpy.array(error_robust)
        std_robust = numpy.array(std_robust)
    error_MV = numpy.array(error_MV)
    std_MV = numpy.array(std_MV)
    error_thresh = numpy.array(error_thresh)
    std_thresh = numpy.std(std_thresh)

    #plots
    fig, axs = plt.subplots(figsize=(10, 8))
    axs.plot(epsilons, error_classical, color='r')
    axs.fill_between(epsilons, (error_classical-std_classical), (error_classical+std_classical), color='r', alpha=.1)
    if add_robust:
        axs.plot(epsilons, error_robust, color='g')
        axs.fill_between(epsilons, (error_robust-std_robust), (error_robust+std_robust), color='g', alpha=.1)
    axs.plot(epsilons, error_MV, color='b')
    axs.fill_between(epsilons, (error_MV-std_MV), (error_MV+std_MV), color='b', alpha=.1)
    axs.plot(epsilons, error_thresh, 'c')
    axs.fill_between(epsilons, (error_thresh-std_thresh), (error_thresh+std_thresh), color='c', alpha=.1)
    axs.set_xlabel("Probability that a cell is contaminated")
    axs.set_ylabel("Frobenius error")
    if add_robust:
        axs.legend(["classical estimator", "Robust estimator", "MV estimator", "MV estimator with thresholding"])
    else:
        axs.legend(["classical estimator", "MV estimator", "MV estimator with thresholding"])
    if eigen_index is not None:
        axs.set_title("Frobenius error on {}th projector".format(eigen_index))
    else:
        axs.set_title("Forbenius error on covariance matrix estimations")
    plt.show()
