import numpy as np
from scipy.stats.qmc import Halton as ghalton


class SimParamSet():
    def __init__(self, normalized=True):
        self.normalized = normalized
        self.params = {}

    def add(self, simparam):
        if simparam.name in self.params.keys():
            raise("parameter name {} already exists in SimParamSet".format(simparam.name))
        self.params[simparam.name] = simparam.dist

    def num_params(self):
        return len(self.params)

    def param_names(self):
        return list(self.params.keys())

    def mean(self):
        m = self.num_params()
        params = self.params
        q_mean = np.zeros([m,1])
        for i, dist in enumerate(params.values()):
            q_mean[i] = dist.mean()

    def var(self):
        m = self.num_params()
        params = self.params
        var = np.zeros([m, 1])
        for i, dist in enumerate(params.values()):
            var[i] = dist.var()

    def pdf(self, q):
        m = self.num_params()
        assert(q.shape[0] == m)
        n = q.shape[1]

        params = self.params
        p_q = np.ones([1, n])
        for i, dist in enumerate(params.values()):
            p_q = p_q * dist.pdf(q[i])
        return p_q

    def logpdf(self, q):
        m = self.num_params()
        assert(q.shape[0] == m)
        n = q.shape[1]

        params = self.params
        p_q = np.zeros([1, n])
        for i, dist in enumerate(params.values()):
            p_q = p_q + np.log(dist.pdf(q[i]))
        return p_q

    def cdf(self, q):
        m = self.num_params()
        assert (q.shape[0] == m)
        n = q.shape[1]

        params = self.params
        p_q = np.empty([m, n])
        for i, dist in enumerate(params.values()):
            p_q[i,:] = dist.cdf(q[i,:])

    def get_gpc_syschars(self):
        params = self.params
        syschar = ''
        for i, dist in enumerate(params.values()):
            syschar = syschar + dist.get_base_dist().orth_polysys_syschar(True)
        return syschar

    def germ2params(self, xi_i_k):
        m = self.num_params()
        assert (xi_i_k.shape[0] == m)
        n = xi_i_k.shape[1]

        params = self.params
        q_i_k = np.zeros([m, n])
        for i, dist in enumerate(params.values()):
            q_i_k[i, :] = dist.base2dist(xi_i_k[i,:])
        return q_i_k

    def params2germ(self, q_i_k):
        m = self.num_params()
        assert (q_i_k.shape[0] == m)
        n = q_i_k.shape[1]

        params = self.params
        xi_i_k = np.zeros([m, n])
        for i, dist in enumerate(params.values()):
            xi_i_k[i, :] = dist.dist2base(q_i_k[i,:])
        return xi_i_k

    def sample(self, n, method = 'MC', **kwargs):
        m = self.num_params()

        q_i_k = np.zeros([m, n])
        params = self.params
        if method == 'MC':
            xi = np.random.rand(m,n)
        elif method == 'QMC':
            gen = ghalton.Halton(m)
            xi = np.array(gen.get(n)).T  # dummy generation to avoid sample point q = 0
        for i, dist in enumerate(params.values()):
            q_i_k[i,:] = dist.invcdf(xi[i,:])

        return q_i_k


if __name__ == "__main__":
    from vamix.channel_flow.gPCE.distributions import UniformDistribution
    from vamix.channel_flow.gPCE.simparameter import SimParameter
    P1 = SimParameter('p1', UniformDistribution(-2,2))
    P2 = SimParameter('p2', UniformDistribution(-2,2))

    Q = SimParamSet()
    Q.add(P1)
    Q.add(P2)

    print(Q.mean())
    print(Q.pdf(np.array([-3, -2, -1, 0, 1, 2, 3]*2).reshape(2,-1)))
    print(Q.cdf(np.array([-3, -2, -1, 0, 1, 2, 3]*2).reshape(2,-1)))
    print(Q.get_gpc_syschars())
    print(Q.params2germ(np.array([-3, -2, -1, 0, 1, 2, 3]*2).reshape(2,-1)))
    print(Q.germ2params(np.array([-2, -1, -0.5, 0, 0.5, 1, 2]*2).reshape(2,-1)))
    Q.sample(10)






