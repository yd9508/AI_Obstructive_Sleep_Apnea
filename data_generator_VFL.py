from fda import *
import numpy as np
import pickle

def dataGeneratorVFL(numSamples = 1000, num_duplicate = 2):
    """
        Generate the data for simulation study of VFL functional regression
        The generated data are two parts. One is the functional response, it will be saved as a single file
        The second part is functional predictors. It will be saved into seperated files given which passive parties it belongs.

        Args:
            numSamples (int): number of observations to generate

        returns:
            none
        """
    print('Simulating Data...')
    # number of basis functions in basis system
    t = 20
    rangeval = [0, 100]
    basisobj = create_bspline_basis(rangeval, t)
    # number of observations n
    n = numSamples
    # number of total features p 20
    p_total = 20
    betaPar = fdPar(basisobj, 0, 0)

    numworkersseq = [2]

    for hh in range(num_duplicate):
        for k in numworkersseq:
            for l in range(k):
                if(p_total % k != 0):
                    raise f'Please set the number of passive parties as a divisor of the number of total predictors, {[p_total, k]} is not a valid pair.'
                p = p_total // k
                predictors = np.zeros([101, p, n])
                weight = np.zeros([t, p])
                for num_p in range(p):
                    weight[:, num_p] = np.ndarray([t], buffer=np.random.normal(l, 0.5, t )) + np.ndarray([t], buffer=np.random.lognormal(num_p * 0.2, l / 10, t ))

                for num_n in range(n):
                    bbspl2 = bifd(np.linspace(1, pow(t, 2), pow(t, 2)).reshape((t, t)),
                                  create_bspline_basis(rangeval, t),
                                  create_bspline_basis(rangeval, t))
                    bifdbasis = bifdPar(bbspl2, 0, 0, 0, 0)
                    betaList = [betaPar, bifdbasis]

                    for num_p in range(p):
                        temp = fd(weight[:, num_p], basisobj)
                        y = eval_fd([i for i in range(rangeval[0], rangeval[1] + 1)], temp)
                        y = np.transpose(y)
                        X = smooth_basis([i for i in range(rangeval[0], rangeval[1] + 1)], y, basisobj).fd
                        predictors[:, num_p, num_n] = eval_fd([i for i in range(rangeval[0], rangeval[1] + 1)], X).reshape(rangeval[1] + 1)


                predictorLst = []
                for num_p in range(p):
                    temp = smooth_basis([i for i in range(rangeval[0], rangeval[1] + 1)], predictors[:, num_p, :],
                                        basisobj).fd
                    predictorLst.append(temp)

                with open('tmp/predictorLst_' + str(l) + '_' + str(k) + '_' + str(hh), 'wb') as file:
                    pickle.dump(predictorLst, file)

            Bmat = np.ndarray([t, t], buffer=np.random.normal(10, 1, t * t))
            yfdobj = fd(np.zeros([t, n]), basisobj)
            beta0estfd = fd(np.ndarray([t], buffer=np.random.normal(0, 0.1, t)), betaList[0].fd.basisobj)
            beta1estbifd = bifd(Bmat, betaList[1].bifd.sbasis, betaList[1].bifd.tbasis)
            lin = linmodList(beta0estfd=beta0estfd, beta1estbifd=beta1estbifd, yhatfdobj=0)

            for l in range(k):
                with open('tmp/predictorLst_' + str(l) + '_' + str(k) + '_' + str(hh), 'rb') as file:
                    predictorLst = pickle.load(file)
                for num_p in range(p):
                    if num_p in range(p // 4):
                        yfdobj = yfdobj + predit_linmod(lin, predictorLst[num_p])

            yfdobj = yfdobj + fd(np.ndarray([t, n], buffer=np.random.normal(0, 0.1, t * n)), basisobj)

            with open('tmp/yfdobj_' + str(k) + '_' + str(hh), 'wb') as file:
                pickle.dump(yfdobj, file)

    print('Data generated.')
    return 0
