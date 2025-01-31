import numpy as np
from distributions import UniformDistribution
from abc import ABC, abstractmethod

class PolynomialSystem(ABC):

    def evaluate(self, deg, xi):
        k = xi.shape[2]
        p = np.zeros([k, deg+1])
        p[:,0] = 0
        p[:,1] = 1
        r = self.recur_coeff(deg+1)
        for d in range(deg):
            p[:, d+2] = (r[d,0] + xi.transpose() * r[d,2]) * p[:, d+1] - r[:,d]
        y_alpha_j = p[:,1:]
        return y_alpha_j

    def sqnorm(self, n):
        deg = max(n.flatten()) + 1
        r = self.recur_coeff(deg)
        nrm2 = self.sqnorm_by_rc(r)
        nrm2 = np.reshape([nrm2[n+1], len(n)])
        return nrm2

    def sqnorm_by_rc(self, rc):
        b = rc[:, 1]
        h = b[0] / b[1:]
        c = rc[1:, 2]
        nrm2 = np.concatenate(np.ones([1]),  h.flatten * np.cumprod(c.flatten()))
        return nrm2

    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @abstractmethod
    def weighting_dist(self):
        pass

    @abstractmethod
    def recur_coeff(self, deg):
        pass


class NormalizedPolynomials(PolynomialSystem):
    def __init__(self, base_polysys):
        self.base_polysys = base_polysys

    def recur_coeff(self, deg):
        r = self.base_polysys.recur_coeff(deg)
        n = np.array(range(deg))
        z = np.concatenate((np.zeros([1]), np.sqrt(self.base_polysys.sqnorm(np.arange(0,deg+1)))), axis=0)
        r = np.array([r[:, 0]*z[n + 1] / z[n + 2],
            r[:, 1] * z[n + 1] / z[n + 2],
            r[:, 2] * z[n] / z[n + 2]])
        return r.transpose()

    def weighting_dist(self):
        dist = self.base_polysys.weighting_dist()
        return dist

class LegendrePolynomials(PolynomialSystem):
    # def __init__(self):

    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        n = np.array(range(deg)).reshape(-1,1)
        zer = np.zeros(n.shape).reshape(-1,1)
        r = np.concatenate((zer, (2*n+1)/(n+1), n/(n+1)), axis=1)
        return r

    @staticmethod
    def sqnorm(n):
        nrm2 = 1/(2*n + 1)
        return nrm2

    @staticmethod
    def weighting_dist():
        dist = UniformDistribution(-1,1)
        return dist

class HermitePolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        #dist = UniformDistribution(-1, 1)
        dist = []
        return dist

class ChebyshevTPolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist


class ChebyshevUPolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist

class LaguerrePolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist


class Monomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist

if __name__ == "__main__":
    LegendrePolynomials.recur_coeff(4)
    LegendrePolynomials.normalized()
    LegendrePolynomials.sqnorm(4)
    LegendrePolynomials.sqnorm(3)
    LegendrePolynomials.normalized().recur_coeff(4)




