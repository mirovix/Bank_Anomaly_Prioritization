#!/usr/bin/env python3

"""
@Author: Miro
@Date: 17/06/2022
@Version: 1.1
@Objective: models configuration
@TODO:
"""

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from configs import train_config as tc


def compile_model_rf():
    return tc.rf


def compile_model_gbc():
    return tc.gbc


def compile_model_voting():
    model = VotingClassifier(estimators=tc.voting_estimators,
                             voting=tc.voting,
                             verbose=tc.verbose,
                             n_jobs=tc.n_jobs,
                             weights=tc.weights)
    return model


def random_grid():
    print(">> parameters to search ", tc.random_grid)
    print("\n")
    return tc.random_grid


def random_search_training(score=tc.score_rs):
    model = VotingClassifier(estimators=tc.voting_estimators_rs, voting=tc.voting)
    rnd_search_cv = RandomizedSearchCV(estimator=model,
                                       param_distributions=random_grid(),
                                       scoring=score,
                                       n_iter=tc.n_iter_rs,
                                       cv=tc.cv_rs,
                                       verbose=tc.verbose,
                                       n_jobs=tc.n_jobs,
                                       return_train_score=True)
    return rnd_search_cv
