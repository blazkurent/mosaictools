# from .polysys import LegendrePolynomials
import numpy as np
from scipy.stats.qmc import Halton as ghalton

# class Distributin():
#     def sample(self, n):
#         xi = np.random.rand(n)
#         return self.invcdf(xi)


class UniformDistribution():
    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b

    def __repr__(self):
        return 'U({}, {})'.format(self.a, self.b)

    def pdf(self, x):
        a = self.a
        b = self.b
        y = 1 / (b - a) * np.ones(x.shape)
        y[x < a] = 0
        y[x > b] = 0
        return y

    def cdf(self, x):
        a = self.a
        b = self.b
        y = (x - a) / (b - a)
        y[x < a] = 0
        y[x > b] = 1
        return y

    def invcdf(self, y):
        a = self.a
        b = self.b
        x = np.full(y.shape, np.nan)
        ind = (y >= 0) & (y <= 1)

        x[ind] = a + (b - a) * y[ind]
        return x

    def sample(self, n, method='MC'):
        if method == 'MC':
            xi = np.random.rand(n)
        elif method == 'QMC':
            gen = ghalton.Halton(1)
            xi = np.array(gen.get(n))  # dummy generation to avoid sample point q = 0
        return self.invcdf(xi)

    def moments(self):
        return self.mean(), self.var(), self.skew(), self.kurt()

    def mean(self):
        return 0.5 * (self.a + self.b)

    def var(self):
        return (self.b - self.a) ** 2 / 12

    def skew(self):
        return 0

    def kurt(self):
        return -6 / 5

    def translate(self, shift, scale):
        m = (self.a + self.b) / 2
        v = scale * (self.b - self.a) / 2

        self.a = m + shift - v
        self.b = m + shift + v

    def get_base_dist(self):
        dist_germ = UniformDistribution(-1, 1)
        return dist_germ

    def base2dist(self, y):
        return self.mean() + y * (self.b - self.a) / 2

    def dist2base(self, x):
        return (x - self.mean()) * 2 / (self.b - self.a)

    # def orth_polysys(self, normalized):
    #     if self.a == -1 & self.b == 1:
    #         if normalized:
    #             polysys = LegendrePolynomials()
    #         else:
    #             polysys = LegendrePolynomials().normalized()
    #     else:
    #         polysys = []
    #     return polysys

    def orth_polysys_syschar(self, normalized):
        if self.a == -1 and self.b == 1:
            if normalized:
                polysys_char = 'p'
            else:
                polysys_char = 'P'
        else:
            polysys_char = []
        return polysys_char


if __name__ == "__main__":
    dist = UniformDistribution(-2,2)
    print(dist.moments())
    print(dist.pdf(np.array([-3,-2,-1,0,1,2,3])))
    print(dist.cdf(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(dist.get_base_dist().a, dist.get_base_dist().b)
    print(dist.dist2base(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(dist.base2dist(np.array([-2, -1, -0.5, 0, 0.5, 1, 2])))



