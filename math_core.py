from typing import Tuple
import numpy as np
import scipy.stats as stats

# Глобальні математичні та статистичні константи
DEFAULT_ALPHA_SIG = 0.05
ZERO_PROTECTION_THRESHOLD = 1e-8
IDEAL_DATA_THRESHOLD = 1e-12


def apply_pca(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Виконує Метод Головних Компонент (МГК/PCA).

    Повертає центровані дані, власні значення, власні вектори та вектор середніх.
    """
    mean_vec = np.mean(data, axis=0)
    centered_data = data - mean_vec
    
    # Коваріаційна матриця
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # Власні числа та вектори
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Сортування за спаданням власних чисел для максимізації дисперсії
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return centered_data, eigenvalues, eigenvectors, mean_vec


def calculate_svd_normal(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Виконує Сингулярний Розклад (SVD) для знаходження вектора нормалі площини.

    Повертає нормаль (вектор), що найкраще апроксимує дані, та вектор середніх значень.
    """
    mean_vec = np.mean(data, axis=0)
    centered_data = data - mean_vec
    
    # SVD розклад простору ознак
    _, _, vt = np.linalg.svd(centered_data, full_matrices=False)
    
    # Нормаль відповідає останньому рядку Vt (мінімальне сингулярне значення)
    normal = vt[-1, :]
    
    # Робимо останню компоненту (Z) додатною для консистентності знаків площин
    if normal[-1] < 0:
        normal = -normal
        
    return normal, mean_vec


def get_plane_equation_coefs(normal: np.ndarray, mean_vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """Перетворює вектор нормалі та точку у коефіцієнти рівняння лінійної площини.

    Повертає кутові коефіцієнти регресії та вільний член для моделі виду:
    Z = c0 + c1*X1 + c2*X2 + ... де остання координата вважається Z.
    """
    z_coef = normal[-1]
    
    # Якщо площина строго вертикальна, уникаємо ділення на нуль
    if abs(z_coef) < ZERO_PROTECTION_THRESHOLD:
        z_coef = ZERO_PROTECTION_THRESHOLD if z_coef >= 0 else -ZERO_PROTECTION_THRESHOLD
        
    # Коефіцієнти при незалежних змінних X (перенос знаків на інший бік)
    coefs = -normal[:-1] / z_coef
    intercept = mean_vec[-1] - np.dot(coefs, mean_vec[:-1])
    
    return coefs, intercept


def alpha_t_test(
    original_data: np.ndarray, 
    distorted_data: np.ndarray, 
    alpha_pred: np.ndarray, 
    alpha_true: np.ndarray, 
    alpha_sig: float = DEFAULT_ALPHA_SIG
) -> str:
    """Проводить парний T-test Стьюдента для перевірки гіпотези відновлення початкової площини (α)."""
    n_features = distorted_data.shape[1]
    divider = "-" * 60
    
    result_str = (
        "=== T-TEST ДЛЯ ВІДНОВЛЕННЯ α ===\n"
        f"Рівень значущості: {alpha_sig}\n\n"
        f"{divider}\n"
        f"{'Парам.':<8} {'α_true':<8} {'α_pred':<8} {'t':<10} {'p-value':<12} {'H0'}\n"
        f"{divider}\n"
    )
    
    for j in range(n_features):
        # Математичне обернене відновлення за допомогою дужок замість похилої риски
        restored = np.sign(distorted_data[:, j]) * (
            (np.abs(distorted_data[:, j]) + ZERO_PROTECTION_THRESHOLD) ** (1.0 / alpha_pred[j])
        )
        
        true_vals = original_data[:, j]
        errors = restored - true_vals
        
        mean_err = np.mean(errors)
        std_err = np.std(errors, ddof=1)
        n = len(errors)
        
        if std_err == 0:
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat = mean_err / (std_err / np.sqrt(n))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
            
        accept = "Так" if p_val > alpha_sig else "Ні"
        result_str += f"α{j+1:<7} {alpha_true[j]:<8.4f} {alpha_pred[j]:<8.4f} {t_stat:<10.3f} {p_val:<12.3f} {accept}\n"
        
    result_str += (
        f"{divider}\n"
        "де 'Так' означає:\n"
        "Не виявлено статистично значущої різниці між відновленими\n"
        f"та істинними даними (при α_sig = {alpha_sig})\n\n"
    )
    
    return result_str


def proper_parameter_t_test(
    true_coefs: np.ndarray, 
    pred_coefs: np.ndarray, 
    X_data: np.ndarray, 
    Z_true: np.ndarray, 
    Z_pred: np.ndarray, 
    alpha_sig: float = DEFAULT_ALPHA_SIG
) -> str:
    """Оцінює статистичну значущість відхилень між теоретичними та знайденими коефіцієнтами регресії."""
    n_points = len(Z_true)
    k_params = len(pred_coefs)
    df = n_points - k_params
    
    # Дисперсія залишків регресії
    residuals = Z_true - Z_pred
    mse_residuals = np.sum(residuals**2) / df
    
    # Розрахунок інформаційної матриці (X^T * X)^-1
    X_T_X_inv = np.linalg.pinv(X_data.T @ X_data)
    
    # Стандартна похибка для кожного оціненого коефіцієнта (SE)
    se_coefs = np.sqrt(np.diagonal(X_T_X_inv) * mse_residuals)
    
    header_divider = "=" * 70
    row_divider = "-" * 70
    
    result = (
        f"{header_divider}\n"
        " T-TEST ПАРАМЕТРІВ (Оцінка допустимості відхилень)\n"
        f"{header_divider}\n"
        f"Рівень значущості: {alpha_sig}\n"
        f"{row_divider}\n"
        f"{'Парам.':<8}{'a_true':<12}{'a_pred':<12}{'Похибка(SE)':<15}{'t':<8}{'p-value':<10}{'H0'}\n"
        f"{row_divider}\n"
    )
    
    for i in range(k_params):
        a_true = true_coefs[i]
        a_pred = pred_coefs[i]
        se = se_coefs[i]
        
        # Захисний механізм для абсолютно детермінованих даних (без дисперсії)
        if se < IDEAL_DATA_THRESHOLD:
            t_stat = 0.0 if np.isclose(a_true, a_pred) else 999.9
            p_val = 1.0 if t_stat == 0.0 else 0.0
        else:
            t_stat = (a_pred - a_true) / se
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
            
        h0_accepted = "Так" if p_val >= alpha_sig else "Ні"
        result += f"a{i+1:<7}{a_true:<12.4f}{a_pred:<12.4f}{se:<15.4f}{t_stat:<8.3f}{p_val:<10.3f}{h0_accepted}\n"
        
    result += (
        f"{row_divider}\n"
        "де 'Так' означає, що знайдене значення лежить у межах допустимої\n"
        "статистичної похибки (різниця не критична).\n\n"
    )
    
    return result