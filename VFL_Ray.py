from fda import *
import ray
import time
import sys

ray.shutdown()
ray.init()
@ray.remote
def linLstGenerate(i_xk, resid, betaL):
    lin = linmod(i_xk, resid, betaL)
    sse = inprod(((predit_linmod(lin, i_xk) - resid) * (predit_linmod(lin, i_xk) - resid))).sum()
    return [lin, sse]


def verticalFederatedFunctionalGradBoostRay(x, y, betaList, boost_control, step_length, epsilon, delta, DP=True,
                                         Clipping=True):
    """
        This function will perform functional gradient boosting algorithm under the framework of vertical federated learning.
        This function uses parallel computing tool: Ray.

        Args:
            x (list): a list with length equals to the number of all predictors
                                    Put all the predictors across all passive parties
                                    the index of this list refers to a specific predictors
                                    each elements is a fd object
            y (list): a list with length 1.
                                  Contains the functional response, the coefficient matrix is of dimension number of basis functions by number of training set
            betaList (int):  a list of basis system for function linmod
            boost_control (int): number of maximum iterations of gradient boosting
            step_length (float): step length of gradient boosting
            epsilon (float): parameter for differential privacy, small value presents strict privacy requirement, takes value from 0 to inif, commonly smaller than 5
            delta (float): parameter for differential privacy, small value presents strict privacy requirement, takes value from 0 to 1.25
            DP (boolean): boolean to control whether to perform differential privacy
            Clipping (boolean): boolean to control whether to perform clipping, to have a limited sensitivity

        returns:
            res (list): a list of linmod objects, store the functional regression results of each iterations
            sseseq (list): a list of floats, store the sum of square errors of each iterations

        """
    numsamples = y[0].coef.shape[1]
    numbasis = y[0].coef.shape[0]
    numPredictors = len(x)
    init = y[0].mean()
    for i in range(1, numsamples):
        init.coef = np.append(init.coef, y[0].mean().coef, axis=1)

    res = []
    res.append(init)
    sseseq = np.zeros(boost_control - 1)
    residual = y[0] - init * 0.5
    if Clipping:
        maxResidual = np.max(np.absolute(residual.coef))
        residual.coef = residual.coef / maxResidual

    if DP:
        if delta >= 1.25 or delta <= 0:
            raise f"Invalid delta value {delta}, should take value from 0 to 1.25. "
        if epsilon <= 0:
            raise f"Invalid epsilon value {epsilon}, should be positive."
        sd = 2 * np.sqrt(numbasis * 2 * np.log(1.25 / delta)) / epsilon
        residual.coef = residual.coef + np.ndarray([numbasis, numsamples],
                                                   buffer=np.random.normal(0, sd, numsamples * numbasis))

    num_iteration = 2
    while num_iteration <= boost_control:

        # Active Party broadcasts the negative gradient to each passive parties
        # Passive parties' side
        tempLst = [linLstGenerate.remote(x[i], residual, betaList) for i in range(numPredictors)]
        tempLst = ray.get(tempLst)

        # Active Party's Side
        # Select best base learner
        sse = np.zeros(numPredictors)
        for num_p in range(numPredictors):
            sse[num_p] = tempLst[num_p][1]

        best = np.argmin(sse)
        # print(f' Round {num_iteration} : {best} selected.')

        # Setup the dynamic step_length by the clipping amount if step_length is null
        if Clipping:
            step_length = max(maxResidual * 0.05, step_length)
        elif step_length is None:
            step_length = 0.05

        # Mark the best base learner
        tempLst[best][0].id = best
        tempLst[best][0].step_length = step_length
        res.append(tempLst[best][0])

        # Re-compute residual by current model and given step_length
        residual = y[0] - pred_gradboostVFL1(res)
        sseseq[num_iteration - 2] = inprod(residual * residual).sum()
        if Clipping:
            maxResidual = np.max(np.absolute(residual.coef))
            residual.coef = residual.coef / maxResidual

        if DP:
            sd = 2 * np.sqrt(numbasis * 2 * np.log(1.25 / delta)) / epsilon
            residual.coef = residual.coef + np.ndarray([numbasis, numsamples],
                                                       buffer=np.random.normal(0, sd, numsamples * numbasis))

        num_iteration = num_iteration + 1

    return res, sseseq