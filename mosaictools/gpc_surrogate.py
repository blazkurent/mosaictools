import numpy as np
from multiindex import *
from gpc_functions import syschar_to_polysys

# ##########################################################################################
#                           GPC SURROGATE MODEL
# ##########################################################################################
class GpcSurrogateModel:
    def __init__(self, Q, p=0, I="default", full_tensor=False, **kwargs):
        self.basis = GpcBasis(Q, p=p, I="default", full_tensor=False)
        self.Q = Q
        self.u_i_alpha = []

    def __repr__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())

    def compute_coeffs_by_regression(self, q_j_k, u_i_k):
        xi_j_k = self.Q.params2germ(q_j_k)
        phi_alpha_k = self.basis.evaluate(xi_j_k)
        u_i_alpha = np.matmul(u_i_k, np.linalg.pinv(phi_alpha_k))
        self.u_i_alpha = u_i_alpha

    def compute_coeffs_by_projection(self, q_j_k, u_i_k, w_k):
        xi_j_k = self.Q.params2germ(q_j_k)
        phi_alpha_k = self.basis.evaluate(xi_j_k)
        u_i_alpha = np.matmul(u_i_k, np.diag(w_k), phi_alpha_k.transpose())
        self.u_i_alpha = u_i_alpha

    def predict_response(self, q_j_k):
        xi_j_k = self.Q.params2germ(q_j_k)
        phi_alpha_k = self.basis.evaluate(xi_j_k)
        u_i_j = np.matmul(self.u_i_alpha, phi_alpha_k)
        return u_i_j



# ##########################################################################################
#                           GPC BASIS
# ##########################################################################################
class GpcBasis:
    # ---------------------Initialization---------------------------------------------------
    def __init__(self, Q, p=0, I="default", full_tensor=False, **kwargs):
        m = Q.num_params()
        self.m = m

        self.syschars = Q.get_gpc_syschars()
        self.p = p

        if I == "default":
            self.I = multiindex(self.m, p, full_tensor=full_tensor)
        else:
            self.I = I

    # ---------------------set how gpc looks when printed ---------------------------------------------------
    def __repr__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())

    # ----------------------------------------- size --------------------------------------------------
    def size(self):
        return [self.I.shape]

    # ----------------------------Evaluate basis functions ---------------------------------------------------
    def evaluate(self, xi, dual=False):
        syschars = self.syschars
        I = self.I
        m = self.m
        M = self.I.shape[0]
        if xi.ndim == 1:
            xi = xi.reshape(-1, 1)
        k = xi.shape[1]
        deg = max(self.I.flatten())

        p = np.zeros([m, k, deg + 2])
        p[:, :, 0] = np.zeros(xi.shape)
        p[:, :, 1] = np.ones(xi.shape)

        if len(syschars) == 1:
            polysys = syschar_to_polysys(syschars)
            r = polysys.recur_coeff(syschars, deg)
            for d in range(deg):
                p[:, :, d + 2] = (r[d, 0] + xi * r[d, 1]) * p[:, :, d + 1] - r[d, 2] * p[:, :, d]
        else:
            for j, syschar in enumerate(syschars):
                polysys = syschar_to_polysys(syschar)
                r = polysys.recur_coeff(deg)
                for d in range(deg):
                    p[j, :, d + 2] = (r[d, 0] + xi[j, :] * r[d, 1]) * p[j, :, d + 1] - r[d, 2] * p[j, :, d]

        y_alpha_j = np.ones([M, k])
        for j in range(m):
            y_alpha_j = y_alpha_j * p[j, :, I[:, j] + 1]

        if dual:
            nrm2 = self.norm(do_sqrt=False)
            y_alpha_j = (y_alpha_j / nrm2.reshape(-1, 1)).transpose()
        return y_alpha_j

    # ------------------------Compute the norm of the basis functions-----------------------
    def norm(self, do_sqrt=True):
        syschars = self.syschars
        I = self.I
        m = self.m
        M = self.I.shape[0]

        if syschars == syschars.lower():
            norm_I = np.ones([M, 1])
            return norm_I

        if len(syschars) == 1:
            # max degree of univariate polynomials
            deg = max(self.I.flatten())
            polysys = syschar_to_polysys(syschars)
            nrm = polysys.sqnorm(range(deg + 1))
            norm2_I = np.prod(nrm[I].reshape(I.shape), axis=1)

        else:
            norm2_I = np.ones([M])
            for j in range(m):
                deg = max(I[:, j])
                polysys = syschar_to_polysys(syschars[j])
                nrm2 = polysys.sqnorm(np.arange(deg + 1))
                norm2_I = norm2_I * nrm2[I[:, j]]
        if do_sqrt:
            norm_I = np.sqrt(norm2_I)
        else:
            norm_I = norm2_I

        return norm_I


# ##########################################################################################
#                           UTILS
# ##########################################################################################

#
# ##########################################################################################
#                           TEST
# ##########################################################################################
def main():
    print(multiindex(3, 4))
    gPCE = GpcSurrogateModel('PP', p=3)
    gPCE.basis.norm()
    print(gPCE.basis.evaluate(np.array([np.arange(-1, 1, 0.1)] * 2)))
    print(gPCE)


if __name__ == "__main__":
    main()
