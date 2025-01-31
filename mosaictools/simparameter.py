import numpy as np
from .gpc_functions import syschar_to_polysys

class SimParameter():
    def __init__(self, name, dist, **kwargs):
        self.name = name
        self.dist = dist

    def __repr__(self):
        return 'Param({}, {})'.format(self.name, self.dist.__repr__())

    def mean(self):
        return self.dist.mean()

    def var(self):
        return self.dist.var()

    def moments(self):
        return self.dist.moments()

    def pdf(self, x):
        return self.dist.pdf(x)

    def logpdf(self, x):
        return np.log(self.dist.pdf(x))

    def cdf(self, p):
        return self.dist.cdf(p)

    def sample(self, n):
        return self.dist.sample(n)

    def get_gpc_dist(self):
        return self.dist.get_base_dist()

    def get_gpc_polysys(self, normalized):
        syschar = self.get_gpc_syschar(normalized)
        return syschar_to_polysys(syschar)

    def get_gpc_syschar(self, normalized):
        return self.dist.get_base_dist().orth_polysys_syschar(normalized)

    def germ2param(self,x):
        q = self.dist.base2dist(x)
        return q

    def param2germ(self, q):
        x = self.dist.dist2base(q)
        return x

if __name__ == "__main__":
    from vamix.channel_flow.gPCE import UniformDistribution
    P = SimParameter('p', UniformDistribution(-2,2))
    print(P.moments())
    print(P.pdf(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(P.cdf(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(P.get_gpc_polysys(True))
    print(P.param2germ(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(P.germ2param(np.array([-2, -1, -0.5, 0, 0.5, 1, 2])))
    print(P.get_gpc_syschar(False))
