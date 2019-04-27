from random import randint
import numpy as np


class SupportVectorClassifier(object):

    def __init__(self, C, max_iterations, tolerance):
        self.C = C
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, dimension = X.shape

        self.w = np.zeros(dimension)
        self.b = 0

        alpha = np.zeros(samples)
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            alpha_old = np.copy(alpha)
            # Можно использовать эвристику для выбора
            # более подходящего множителя для оптимизации
            # для увеличения сходимости
            for i2 in range(samples):
                # Опять же существуют эвристики для подбора второго множителя
                # Выбираем случайным образом
                i1 = self._get_random_sample(samples, i2)
                x1, x2 = X[i1], X[i2]
                y1, y2 = y[i1], y[i2]

                k11 = self._linear_kernel(x1, x1)
                k22 = self._linear_kernel(x2, x2)
                k12 = self._linear_kernel(x1, x2)
                eta = k11 + k22 - 2 * k12

                if eta == 0:
                    continue

                a1, a2 = alpha[i1], alpha[i2]
                # Минимальное и максимальное значение для a2
                lower = self._calculate_lower(y1, y2, a1, a2)
                upper = self._calculate_upper(y1, y2, a1, a2)
                # Пересчет весов
                self.w = self._calculate_w(X, y, alpha)
                self.b = self._calculate_b(X, y)
                # Подсчет ошибок
                e1 = self._calculate_error(x1, y1)
                e2 = self._calculate_error(x2, y2)
                # Пересчет множителей Лагранжа
                alpha[i2] = a2 + y2 * (e1 - e2) / eta
                alpha[i2] = max(lower, alpha[i2])
                alpha[i2] = min(upper, alpha[i2])
                alpha[i1] = a1 + y1 * y2 * (a2 - alpha[i2])
            # Условие выхода - малое изменение вектора alpha
            delta_alpha = alpha - alpha_old
            if np.dot(delta_alpha, delta_alpha) < self.tolerance:
                break
        # Алгоритм возможно не сошелся
        if iterations == self.max_iterations:
            print("WARNING: performed %d iterations!" % self.max_iterations)
        # Окончательный подсчет весов
        self.w = self._calculate_w(X, y, alpha)
        self.b = self._calculate_b(X, y)

    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T) - self.b).astype(np.int8)

    def _calculate_w(self, X, y, alpha):
        return np.dot(X.T, y * alpha)

    def _calculate_b(self, X, y):
        # Для большей устойчивости берется среднее
        return np.mean(np.dot(self.w.T, X.T) - y)

    def _calculate_lower(self, y1, y2, a1, a2):
        if y1 == y2:
            return max(0, a1 + a2 - self.C)
        else:
            return max(0, a2 - a1)

    def _calculate_upper(self, y1, y2, a1, a2):
        if y1 == y2:
            return min(self.C, a1 + a2)
        else:
            return min(self.C, self.C + a2 - a1)

    def _calculate_error(self, x, y):
        return np.sign(np.dot(self.w.T, x) - self.b) - y

    def _get_random_sample(self, samples, current_sample):
        another_sample = current_sample
        while another_sample == current_sample:
            another_sample = randint(0, samples - 1)
        return another_sample

    def _linear_kernel(self, x1, x2):
        return np.dot(x1.T, x2)
