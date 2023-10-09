import numpy as np
import scipy

overlap_matrix = np.load("overlap.npy")
H = np.load("core_hamiltonian.npy")
int2e = np.load("int2e.npy")
nao = len(overlap_matrix[0])
assert nao == len(int2e[0])
assert int2e.shape == (nao, nao, nao, nao)
dm = np.eye(nao)

# Calculate the coulomb matrices from density matrix
def get_j(dm):
    J = np.zeros((nao, nao))  # Initialize the Coulomb matrix

    # Loop over all indices of the Coulomb matrix
    for p in range(nao):
        for q in range(nao):
            # Calculate the Coulomb integral for indices (p,q)
            for r in range(nao):
                for s in range(nao):
                    J[p, q] += dm[r, s] * int2e[p, q, r, s]

    return J

# Calculate the exchange  matrices from density matrix
def get_k(dm):
    K = np.zeros((nao, nao))  # Initialize the K matrix

    # Loop over all indices of the K matrix
    for p in range(nao):
        for q in range(nao):
            # Calculate the K integral for indices (p,q)
            for r in range(nao):
                for s in range(nao):
                    K[p, q] += dm[r, s] * int2e[p, r, q, s]

    return K

# Calculate the density matrix
def get_dm(fock, nocc):
    dm = np.zeros((nao, nao))
    S = overlap_matrix
    A = scipy.linalg.fractional_matrix_power(S, -0.5)
    F_p = A.T @ fock @ A
    eigs, coeffsm = np.linalg.eigh(F_p)

    c_occ = A @ coeffsm
    c_occ = c_occ[:, :nocc]
    for i in range(nocc):
        for p in range(nao):
            for q in range(nao):
                dm[p, q] += c_occ[p, i] * c_occ[q, i]
    return dm
# Maximum SCF iterations
max_iter = 100
E_conv = 1.0e-10
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0
for scf_iter in range(1, max_iter + 1):
    # GET Fock martix
    F = H + 2 * get_j(dm) - get_k(dm)
    assert F.shape == (nao, nao)

    SCF_E = np.sum(np.multiply((H + F), dm))
    dE = SCF_E - E_old
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E' % (scf_iter, SCF_E, dE))

    if (abs(dE) < E_conv):
        print("SCF convergence! Congrats")
        break
    E_old = SCF_E

    dm = get_dm(F, 5)

assert(np.abs(SCF_E + 84.1513215474753) < 1.0e-10)