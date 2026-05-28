"""
@file        svm_benchmark.py
@author      Lucía Parreño Legorburo
@brief       Picklable SVM objective function for PSO hyperparameter optimisation, it is 
                defined as a class at module level so multiprocessing can serialize it
"""


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

Vector = list[float]

SVM_BOUNDS = [(-3.0, 3.0), (-4.0, 1.0)]


class SVMObjective:
    def __init__(self, X, y, cv: int = 5):
        """
        Pickable callable for SVM cross-validation dividing the dataset in equal parts to train and test, 
        it is defined at a module level so that V2 can serizalize it.
        Args:
            X : features of the dataset
            y : classes of the dataset
            cv (int, optional): cross-validation. Defaults to 5.
        """
        self.X  = X
        self.y  = y
        self.cv = cv

    def __call__(self, x: Vector) -> float:
        """
        Evaluates SVM fitness given a C (margin of separation in classes) and gamma (influence ratio,
        if C is low it will be more tolerant, while being high means it is strict,
        if gamma is low values will be more global, while being high means values are more local.
        The search is done using a log10 space.

        Args:
            x (Vector): vector with the dataset

        Returns:
            float: cross-validation accuracy for the PSO to minimise it
        """

        c = 10 ** x[0]
        gamma = 10 ** x[1]
        pipe  = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(C=c, gamma=gamma, kernel="rbf", random_state=42)),
        ])
        scores = cross_val_score(pipe, self.X, self.y,
                                 cv=self.cv, scoring="accuracy", n_jobs=-1)
        return 1.0 - scores.mean()


class TrackedSVMObjective(SVMObjective):
    """
    Same as SVMObjective but records every evaluated point.

    NOTE: tracking only works for V0, V1, V3.
    V2 multiprocessing runs in separate processes so it can only use SVMObjective.
    """
    def __init__(self, X, y, cv: int = 5):
        super().__init__(X, y, cv)
        self.evaluated_points: list[tuple] = []

    def __call__(self, x: Vector) -> float:
        f = super().__call__(x)
        self.evaluated_points.append((x[0], x[1], 1.0 - f))
        return f


def make_svm_objective(X, y, cv: int = 5) -> SVMObjective:
    """Picklable fitness function for all PSO versions including V2"""
    return SVMObjective(X, y, cv)


def make_tracked_svm_objective(X, y, cv: int = 5) -> TrackedSVMObjective:
    """Tracked fitness function for V0, V1, V3"""
    return TrackedSVMObjective(X, y, cv)