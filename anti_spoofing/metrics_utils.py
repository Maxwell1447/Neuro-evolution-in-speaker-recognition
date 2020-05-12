import numpy as np
from collections import namedtuple
from libmath import pavx

__website__ = "https://gitlab.eurecom.fr/nautsch/pybosaris/tree/master/pybosaris"

__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__credits__ = ["Niko Brummer", "Edward de Villiers", "Anthony Larcher"]
__license__ = "LGPLv3"


Box = namedtuple("Box", "left right top bottom")




def rocch2eer(pmiss, pfa):
    """Calculates the equal error rate (eer) from pmiss and pfa vectors.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.
    Use rocch.m to convert target and non-target scores to pmiss and
    pfa values.

    :param pmiss: the vector of miss probabilities
    :param pfa: the vector of false-alarm probabilities

    :return: the equal error rate
    """
    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        # xx and yy should be sorted:
        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            'pmiss and pfa have to be sorted'

        XY = np.column_stack((xx, yy))
        dd = np.dot(np.array([1, -1]), XY)
        if np.min(np.abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = np.linalg.solve(XY, np.array([[1], [1]]))
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (np.sum(seg))

        eer = max([eer, eerseg])
    return eer


def rocch(tar_scores, nontar_scores):
    """ROCCH: ROC Convex Hull.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.

    For a demonstration that plots ROCCH against ROC for a few cases, just
    type 'rocch' at the MATLAB command line.

    :param tar_scores: vector of target scores
    :param nontar_scores: vector of non-target scores

    :return: a tupple of two vectors: Pmiss, Pfa
    """
    Nt = tar_scores.shape[0]
    Nn = nontar_scores.shape[0]
    N = Nt + Nn
    scores = np.concatenate((tar_scores, nontar_scores))
    # Pideal is the ideal, but non-monotonic posterior
    Pideal = np.concatenate((np.ones(Nt), np.zeros(Nn)))
    #
    # It is important here that scores that are the same
    # (i.e. already in order) should NOT be swapped.rb
    perturb = np.argsort(scores, kind='mergesort')
    #
    Pideal = Pideal[perturb]
    Popt, width, foo = pavx(Pideal)
    #
    nbins = width.shape[0]
    pmiss = np.zeros(nbins + 1)
    pfa = np.zeros(nbins + 1)
    #
    # threshold leftmost: accept everything, miss nothing
    left = 0  # 0 scores to left of threshold
    fa = Nn
    miss = 0
    #
    for i in range(nbins):
        pmiss[i] = miss / Nt
        pfa[i] = fa / Nn
        left = int(left + width[i])
        miss = np.sum(Pideal[:left])
        fa = N - left - np.sum(Pideal[left:])
    #
    pmiss[nbins] = miss / Nt
    pfa[nbins] = fa / Nn
    #
    return pmiss, pfa
