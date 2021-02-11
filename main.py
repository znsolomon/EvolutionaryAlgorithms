import numpy as np


class Data:
    def __init__(self, n, m, w, h, R, P, c_matrix, d_matrix, p_matrix, alpha, T,
                 module_mask, increment_number, external_allocation, mxb, mnb, u_bound, l_bound):
        self.P = P  # enum of teaching preferences (n x m)
        self.n = n  # no. staff (int)
        self.m = m  # no. modules (int)
        self.R = R  # teaching proportions last year (n x m)
        self.l_bound = l_bound#
        self.u_bound = u_bound#
        self.mnb = mnb  # ?? (currently set to n)
        self.mxb = mxb  # ?? (currently set to m)
        self.external_allocation = external_allocation#
        self.increment_number = increment_number#
        self.module_mask = module_mask#
        self.T = T  # boolean array of which staff have taught modules before (n x m)
        self.alpha = alpha  # param - controls effect of T (float)
        self.p_matrix = p_matrix  # prep time of each module (m)
        self.d_matrix = d_matrix  # contact hours of each module (m)
        self.c_matrix = c_matrix  # co-ordinator hours of each module (m)
        self.h = h  # contractual hours of each staff member (n)
        self.w = w  # non-teaching workloads of each staff member (n)


class Solution:
    def __init__(self, x, c):
        self.X = x
        self.C = c


if __name__ == '__main__':
    data = Data()


def swap_crossover(P, data):
    # Performs crossover mutation on P
    k = len(P)
    R_comb = np.random.permutation(k)
    for i in range(0, k-1, 2):
        parent1 = P[R_comb[i]]
        parent2 = P[R_comb[i+1]]
        child1 = parent1
        child2 = parent2
        rand_x = np.random.randint(data.m)
        rand_y = np.random.randint(data.n)
        if np.random.rand() < 0.8:  # 0.8% chance of crossover
            child1.X[rand_x, rand_y] = parent2.X[rand_x, rand_y]
            child1.C[rand_x, rand_y] = parent2.C[rand_x, rand_y]

            child2.X[rand_x, rand_y] = parent1.X[rand_x, rand_y]
            child2.C[rand_x, rand_y] = parent1.C[rand_x, rand_y]
        P[R_comb[i]] = child1
        P[R_comb[i+1]] = child2
    return P


def swap_mutation(P, data):
    max_to_vary = 1  # Just switches on one element - may want to vary

    for i in range(len(P)):
        child = P[i]
        for k in range(max_to_vary):
            rm = np.random.permutation(data.m) # get a module at random
            rm = rm[0]
            if data.module_mask[rm] == 1:
                r = np.random.permutation(data.n)
                child.C[rm,:] = 0
                child.X[rm,:] = 0
                child.C[rm, r[0]] = 1 # swap staff member involved
                child.X[rm, r[0]] = data.increment_number(rm) - data.external_allocation(rm)
            else:
                I = np.where(child.X[rm,:] > 0) # get indicies where teaching is happening
                r = np.random.permutation(len(I))
                I = I[r] # randomly permute
                if len(I) > 0: # some delivery internally
                    if np.random.rand() < 0.5:
                        child.X[rm[0], I[0]] = child.X[rm[0], I[0]] - 1
                        if len(r) < data.module_mask[rm]: # can add extra staff
                            rn = np.random.permutation(data.n) # allocate to a random other
                            child.X[rm[0], rn[0]] = child.X[rm[0], rn[0]] + 1
                            # always assign coordination to staff teaching most of module
                            child.C[rm[0], :]=0
                            index = max(child.X[rm[0], :])
                            child.C[rm[0], index] = 1
                        else: # can only shift between staff
                            child.X[rm[0], I[1]] = child.X[rm[0], I[1]] + 1
                            # always assign coordination to staff teaching most of module
                            child.C[rm[0], :]=0
                            index = max(child.X[rm[0], :])
                            child.C[rm[0], index] = 1

                    else: # randomly remove teaching from one member of staff and give it to another
                        rn = np.random.permutation(data.n) # allocate to a random other
                        if rn[0] == I[0]:
                            rn = rn[1]
                        else:
                            rn = rn[0]
                        child.X[rm[0], rn] = child.X[rm[0], rn] + child.X[rm[0], I[0]]
                        child.X[rm[0], I[0]] = 0
                        child.C[rm[0], rn] = max(child.C[rm[0], I[0]], child.C[rm[0], rn])
                        child.C[rm[0], I[0]] = 0
                else: # where no teaching due ot external delivery swap coordinator
                    child.C[rm[0], :] = 0
                    index = np.random.permutation(data.n)
                    child.C[rm[0], index[0]] = 1

        P[i] = child


    # enforce constraints
    P = teaching_constraints(P, data)
    return P


def swap_random(data):  # Creates a random solution
    x = np.zeros(data.m, data.n)
    c = np.zeros(data.m, data.n)
    for i in range(data.m):
        for j in range(data.n):
            rn = np.random.randint(100)
            x[i, j] = rn
            c[i, j] = rn % 50

    sol = Solution(x, c)
    return sol
