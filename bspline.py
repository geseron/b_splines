import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class BSplineRegression:
    """B-сплайновая регрессия с настраиваемыми узлами."""
    
    def __init__(self, degree=3, knots=None, n_knots=None, regularization=0.0):
        self.degree = degree
        self.knots = knots          # явный вектор внутренних узлов (если задан)
        self.n_knots = n_knots      # количество внутренних узлов (если knots не задан)
        self.regularization = regularization  # коэффициент гребневой регуляризации
        self.coefficients_ = None
        self.knots_full_ = None     # полный вектор узлов (с кратностью на концах)

    def _build_knot_vector(self, x):
        """Строит полный вектор узлов на основе внутренних узлов."""
        if self.knots is None:
            # Если узлы не заданы, создаем равномерные
            t = np.linspace(x.min(), x.max(), self.n_knots + 2)[1:-1]
        else:
            t = self.knots
        # Добавляем кратность degree на концах
        return np.concatenate(([x.min()] * self.degree, t, [x.max()] * self.degree))

    def _basis_matrix(self, x, knots_full):
        """Строит матрицу базисных функций для всех x."""
        n_basis = len(knots_full) - self.degree - 1
        m = len(x)
        mat = np.zeros((m, n_basis))
        for j, xi in enumerate(x):
            for i in range(n_basis):
                mat[j, i] = self._basis_function(i, self.degree, xi, knots_full)
        return mat

    def _basis_function(self, i, p, t, knots):
        """Рекурсивное вычисление базисной функции (как ранее)."""
        if p == 0:
            if knots[i] <= t < knots[i+1]:
                return 1.0
            if t == knots[-1] and i == len(knots)-2:
                return 1.0
            return 0.0
        else:
            left = 0.0
            right = 0.0
            denom1 = knots[i+p] - knots[i]
            if denom1 != 0:
                left = (t - knots[i]) / denom1 * self._basis_function(i, p-1, t, knots)
            denom2 = knots[i+p+1] - knots[i+1]
            if denom2 != 0:
                right = (knots[i+p+1] - t) / denom2 * self._basis_function(i+1, p-1, t, knots)
            return left + right

    def fit(self, X, y):
        """Обучает модель: строит матрицу базисов и решает МНК/гребневую регрессию."""
        X = np.asarray(X).ravel()
        self.knots_full_ = self._build_knot_vector(X)
        self.n_basis_ = len(self.knots_full_) - self.degree - 1
        B = self._basis_matrix(X, self.knots_full_)
        # Гребневая регрессия: (B^T B + λ I) c = B^T y
        if self.regularization > 0:
            lhs = B.T @ B + self.regularization * np.eye(self.n_basis_)
            rhs = B.T @ y
            self.coefficients_ = np.linalg.solve(lhs, rhs)
        else:
            self.coefficients_, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
        return self

    def predict(self, X):
        """Предсказывает значения для новых точек."""
        X = np.asarray(X).ravel()
        B = self._basis_matrix(X, self.knots_full_)
        return B @ self.coefficients_