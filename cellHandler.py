def DDCW(data):
    # 1. drop variables with too many missing values
    
    # 2. DetectDeviatingCells (Rousseuw and Vand den Bossche, 2018)
    # 3. PCA on remaining data
    # 4. wrapped location and covariance (Raymaekers and Rousseuw, 2019) and computation of u_ij
    # 5. removing all cases in Robust Distance exceeds khi2
    # 6. projection of the remaining zi on eigenvectors of sigma and new wrapped location and covariance
    # 7. return to original basis and reverse scaling

class CellHandler:
    def __init__(self):
        self.data = None
        self._covariance = None
        self._mean = None
        
    def find_outliers(self):
        return 0
    
    def estimate_cov(self, outliers=[]):
        return 0
    
    def fit(self, data, steps=10):
        self.data=data
        if self._covariance is None:
            self._mean, self._covariance = self.estimate_cov()
        
        for k in range(steps):
            outliers = self.find_outliers()
            self._mean, self._covariance = self.estimate_cov(outliers=outliers)