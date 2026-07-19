import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.stats as stats
from typing import Any, Dict, List, Optional, Tuple, Callable

# Імпорт математичного ядра проекту
from math_core import (
    apply_pca, 
    calculate_svd_normal, 
    get_plane_equation_coefs, 
    alpha_t_test, 
    proper_parameter_t_test
)

# Глобальні константи для інтерфейсу
DEFAULT_ALPHA_SIG = "0.05"
TEXT_BOX_HEIGHT = 16
REPORT_WINDOW_SIZE = "650x650"
PCA_VARIANCE_THRESHOLD = 95.0
BOXCOX_EPSILON = 1e-5


class Stage3UI:
    """Клас для керування інтерфейсом 3-го етапу: Нормалізація, МГК та векторний аналіз."""

    def __init__(self, frame: tk.Frame, app: Any):
        """Ініціалізує компоненти інтерфейсу користувача та зв'язує їх з головним додатком."""
        self.app = app
        self.frame = frame
        self.normal_calculated_flag: bool = False
        self.calculated_coefs: Optional[np.ndarray] = None
        self.alpha_pred_boxcox: Optional[np.ndarray] = None
        self.distorted_data_backup: Optional[np.ndarray] = None
        self.mean_vec: Optional[np.ndarray] = None

        self._build_ui()

    def _build_ui(self):
        """Внутрішній метод для побудови сітки та віджетів інтерфейсу."""
        # 1. Блок нормалізації Box-Cox
        bc_frame = ttk.LabelFrame(self.frame, text="1. Нормалізація (Box-Cox)")
        bc_frame.pack(fill=tk.X, pady=2)
        
        bc_btns = ttk.Frame(bc_frame)
        bc_btns.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(bc_btns, text="Вирівняти (Box-Cox)", command=self.apply_boxcox).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(bc_btns, text="Візуалізація", command=self.vis_boxcox).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 2. Блок методу головних компонент (МГК)
        mgk_frame = ttk.LabelFrame(self.frame, text="2. Метод Головних Компонент (МГК)")
        mgk_frame.pack(fill=tk.X, pady=2)
        
        mgk_btns = ttk.Frame(mgk_frame)
        mgk_btns.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(mgk_btns, text="МГК (Видалити шум)", command=self.apply_mgk).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(mgk_btns, text="Візуалізація", command=self.vis_mgk).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 3. Блок векторного пошуку нормалі та статистичного аналізу
        slar_frame = ttk.LabelFrame(self.frame, text="3. Векторний пошук нормалі та Аналіз")
        slar_frame.pack(fill=tk.X, pady=2)
        
        sig_frame = ttk.Frame(slar_frame)
        sig_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(sig_frame, text="Рівень значущості (α гіпотези):").pack(side=tk.LEFT)
        self.entry_alpha_sig = ttk.Entry(sig_frame, width=8)
        self.entry_alpha_sig.insert(0, DEFAULT_ALPHA_SIG)
        self.entry_alpha_sig.pack(side=tk.LEFT, padx=5)

        ttk.Button(slar_frame, text="Розв'язати (Векторний підхід SVD)", command=self.calc_normal).pack(fill=tk.X, padx=5, pady=2)
        
        btn_frame = ttk.Frame(slar_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(btn_frame, text="Візуальне порівняння площин", command=self.vis_plane_comparison).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Оцінка точності (Метрики та T-test)", command=self.show_metrics).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Лог-консоль виведення результатів
        self.res_box = tk.Text(self.frame, height=TEXT_BOX_HEIGHT)
        self.res_box.pack(fill=tk.BOTH, expand=True, pady=5)

    def _get_axis_name(self, index: int) -> str:
        """Повертає назву осі з конфігурації додатку або генерує стандартне ім'я X_i."""
        if hasattr(self.app, 'feature_names') and len(self.app.feature_names) > index:
            return self.app.feature_names[index]
        return f"X{index + 1}"

    def apply_boxcox(self):
        """Виконує трансформацію Бокса-Кокса для вирівнювання розподілів ознак."""
        if getattr(self.app, 'stage2_data', None) is None:
            messagebox.showwarning("Увага", "Немає даних для обробки. Виконайте попередній етап.")
            return
            
        data = self.app.stage2_data
        bc_data = np.zeros_like(data)
        num_dimensions = data.shape[1]
        lambdas = []
        
        # Запобігання фрізам SVD шляхом очищення застарілого кешу МГК
        self.app.stage3_mgk_data = None 
        
        for i in range(num_dimensions):
            col_data = data[:, i]
            if np.any(col_data <= 0):
                col_data = col_data - np.min(col_data) + BOXCOX_EPSILON
                
            lmb = stats.boxcox_normmax(col_data)
            lambdas.append(lmb)
            bc_data[:, i] = stats.boxcox(col_data, lmbda=lmb)
            
        self.app.stage3_bc_data = bc_data
        self.alpha_pred_boxcox = np.array([1.0 / l if l != 0 else 1.0 for l in lambdas])
        self.distorted_data_backup = np.copy(data)
        
        # Виведення результатів
        self.res_box.delete(1.0, tk.END)
        self.res_box.insert(tk.END, "--- РЕЗУЛЬТАТИ НОРМАЛІЗАЦІЇ BOX-COX ---\n")
        self.res_box.insert(tk.END, "Для кожної осі знайдено оптимальну λ. Еквівалентне α викривлення ≈ 1/λ\n\n")
        
        for i, lmb in enumerate(lambdas):
            axis_name = self._get_axis_name(i)
            equiv_alpha = 1.0 / lmb if lmb != 0 else 0.0 
            self.res_box.insert(tk.END, f"{axis_name}: оптимальна λ = {lmb:.4f}  =>  (Еквівалентне α ≈ {equiv_alpha:.4f})\n")
            
        self.res_box.see(tk.END)
        messagebox.showinfo("OK", "Box-Cox виконано для всіх вимірів.")

    def apply_mgk(self):
        """Здійснює аналіз головних компонент (PCA) та виконує фільтрацію шумів."""
        source = getattr(self.app, 'stage3_bc_data', None)
        if source is None: 
            source = getattr(self.app, 'stage2_data', None)
        if source is None: 
            return messagebox.showwarning("Увага", "Немає даних.")

        centered_data, eigenvalues, eigenvectors, self.mean_vec = apply_pca(source)

        total_var = np.sum(eigenvalues)
        var_ratio = (eigenvalues / total_var) * 100
        cum_var_ratio = np.cumsum(var_ratio)
        n_cols = len(eigenvalues)

        self.res_box.insert(tk.END, "\n=== РЕЗУЛЬТАТИ МГК ===\n\n")
        header = f"{'':<16}" + "".join([f"x'_{i+1:<7}" for i in range(n_cols)]) + "\n"
        self.res_box.insert(tk.END, header + "-" * len(header) + "\n")
        
        for i in range(n_cols):
            row_str = f"Ознака x{i+1:<8}"
            for j in range(n_cols): 
                row_str += f"{eigenvectors[i, j]:>8.2f}"
            self.res_box.insert(tk.END, row_str + "\n")

        self.res_box.insert(tk.END, "-" * len(header) + "\n")
        self.res_box.insert(tk.END, f"{'Власні числа':<16}" + "".join([f"{v:>8.3f}" for v in eigenvalues]) + "\n")
        self.res_box.insert(tk.END, f"{'% на напрям':<16}" + "".join([f"{v:>8.1f}" for v in var_ratio]) + "\n")
        self.res_box.insert(tk.END, f"{'Накопичений %':<16}" + "".join([f"{v:>8.1f}" for v in cum_var_ratio]) + "\n\n")

        idx_95 = np.argmax(cum_var_ratio >= PCA_VARIANCE_THRESHOLD)
        n_kept = idx_95 + 1
        noise_var = np.sum(var_ratio[n_kept:]) if n_kept < n_cols else 0.0
        
        self.res_box.insert(tk.END, f"Залишаємо {n_kept} головних компонент (накопичено {cum_var_ratio[idx_95]:.2f}%).\n")
        self.res_box.insert(tk.END, f"Відсікаємо кінцеві {noise_var:.2f}% дисперсії як інформаційний шум.\n")

        P = eigenvectors[:, :n_kept]
        projected = centered_data @ P @ P.T
        self.app.stage3_mgk_data = projected + self.mean_vec
        self.res_box.see(tk.END)
        messagebox.showinfo("OK", "МГК виконано. Шум відфільтровано.")

    def calc_normal(self):
        """Знаходить вектор нормалі гіперплощини на основі SVD розкладу найкращих доступних даних."""
        if getattr(self.app, 'stage3_mgk_data', None) is not None:
            data = self.app.stage3_mgk_data
            source_name = "МГК"
        elif getattr(self.app, 'stage3_bc_data', None) is not None:
            data = self.app.stage3_bc_data
            source_name = "Box-Cox"
        elif getattr(self.app, 'stage2_data', None) is not None:
            data = self.app.stage2_data
            source_name = "Етап 2"
        else:
            return messagebox.showwarning("Увага", "Немає даних для розрахунку.")

        normal_vector, mean_vec = calculate_svd_normal(data)
        coefs_raw, _ = get_plane_equation_coefs(normal_vector, mean_vec)
        
        self.calculated_coefs = coefs_raw
        self.normal_calculated_flag = True

        self.res_box.insert(tk.END, f"\n=== ПОШУК НОРМАЛІ (SVD) на базі {source_name} ===\n")
        self.res_box.insert(tk.END, f"Вектор нормалі (N): {np.round(normal_vector, 4)}\n\n")
        
        self.res_box.insert(tk.END, "Знайдені коефіцієнти площини (відлік від оригінальних зсувів):\n")
        for i, c in enumerate(coefs_raw):
            axis_name = self._get_axis_name(i)
            self.res_box.insert(tk.END, f"  a{i+1} ({axis_name}): {c:.4f}\n")
        
        self.res_box.see(tk.END)

    def get_recovered_plane_fn(self) -> Optional[Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]]:
        """Генерує та повертає математичну функцію відновленої площини для 3D графіків."""
        if not self.normal_calculated_flag or self.calculated_coefs is None:
            return None
            
        coefs = self.calculated_coefs
        shifts = getattr(self.app, 'stage1_shifts', np.zeros(3))
        
        def plane_function(x: np.ndarray, y: np.ndarray, idx_x: int, idx_y: int) -> np.ndarray:
            z = shifts[-1]
            if idx_x < len(coefs): 
                z += coefs[idx_x] * (x - shifts[idx_x])
            if idx_y < len(coefs): 
                z += coefs[idx_y] * (y - shifts[idx_y])
            return z
            
        return plane_function

    def vis_boxcox(self):
        """Ініціює візуалізацію даних до та після перетворення Бокса-Кокса."""
        if getattr(self.app, 'stage3_bc_data', None) is None: 
            return messagebox.showwarning("Увага", "Спочатку виконайте Box-Cox.")
        
        self.app.show_plot(
            3,
            {"До Box-Cox": self.app.stage2_data, "Після Box-Cox": self.app.stage3_bc_data},
            "Аналіз Box-Cox",
            show_plane=True
        )

    def vis_mgk(self):
        """Відображає результат фільтрації шуму за допомогою МГК разом із відновленою площиною."""
        if getattr(self.app, 'stage3_mgk_data', None) is None: 
            return messagebox.showwarning("Увага", "Спочатку виконайте МГК.")
            
        source = getattr(self.app, 'stage3_bc_data', None)
        if source is None:
            source = self.app.stage2_data
        
        custom_planes = {}
        recovered_plane = self.get_recovered_plane_fn()
        if recovered_plane is not None:
            custom_planes["МГК + SVD"] = (recovered_plane, 'green')

        self.app.show_plot(
            3,
            {"До МГК": source, "Після МГК (Згладжені)": self.app.stage3_mgk_data},
            "Аналіз: МГК + SVD",
            show_plane=True,
            custom_planes=custom_planes
        )

    def vis_plane_comparison(self):
        """Створює графік для візуального порівняння розрахованої площини з ідеальним орієнтиром."""
        if getattr(self.app, 'stage3_mgk_data', None) is None:
            return messagebox.showwarning("Увага", "Спочатку виконайте МГК.")

        recovered_plane = self.get_recovered_plane_fn()
        if recovered_plane is None:
            return messagebox.showwarning("Увага", "Спочатку розрахуйте вектор нормалі (Крок 3).")

        custom_planes = {
            "Знайдена МГК+SVD": (recovered_plane, 'green')
        }

        self.app.show_plot(
            3,
            {"Очищені дані (МГК)": self.app.stage3_mgk_data},
            "Порівняння знайденої та ідеальної площин",
            show_plane=True, 
            custom_planes=custom_planes
        )

    def show_metrics(self):
        """Аналізує точність знайденого розв'язку: розраховує похибки, R² та викликає T-тести."""
        if not self.normal_calculated_flag or self.calculated_coefs is None:
            return messagebox.showwarning("Увага", "Спочатку розрахуйте вектор нормалі (Крок 3).")
            
        if getattr(self.app, 'stage1_data', None) is None:
            return messagebox.showwarning("Увага", "Немає оригінальних даних з Етапу 1 для порівняння.")
            
        data = self.app.stage1_data
        shifts = getattr(self.app, 'stage1_shifts', np.zeros(data.shape[1]))
        
        X = data[:, :-1]
        Z_true = data[:, -1]
        n_points = len(Z_true)
        
        # Обчислення прогнозних значень
        Z_pred = np.full_like(Z_true, shifts[-1])
        for i, coef in enumerate(self.calculated_coefs):
            Z_pred += coef * (X[:, i] - shifts[i])
            
        residuals = Z_true - Z_pred
        abs_residuals = np.abs(residuals)
        
        # Розрахунок метрик якості регресії
        mse = np.mean(residuals**2)
        mae = np.mean(abs_residuals)
        max_err = np.max(abs_residuals)
        r2 = 1 - (np.sum(residuals**2) / np.sum((Z_true - np.mean(Z_true))**2))
        
        # Перевірка зсуву прогнозів (парний T-критерій Стьюдента)
        t_stat, p_val = stats.ttest_rel(Z_true, Z_pred)
        
        # Відображення вікна звіту
        self._display_report_window(n_points, mse, mae, max_err, r2, X, Z_true, Z_pred, p_val)

    def _display_report_window(self, n_points: int, mse: float, mae: float, max_err: float, r2: float, 
                               X: np.ndarray, Z_true: np.ndarray, Z_pred: np.ndarray, p_val: float):
        """Внутрішній метод для рендерингу окремого вікна детального аналітичного звіту."""
        rep_win = tk.Toplevel(self.frame)
        rep_win.title("Оцінка аналітичного методу (Метрики та Т-тест)")
        rep_win.geometry(REPORT_WINDOW_SIZE)
        
        rep_text = tk.Text(rep_win, wrap=tk.WORD)
        rep_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        rep_text.insert(tk.END, "="*65 + "\n")
        rep_text.insert(tk.END, " ЗВІТ ТОЧНОСТІ АНАЛІТИЧНОГО МЕТОДУ (МГК + SVD)\n")
        rep_text.insert(tk.END, "="*65 + "\n\n")

        # Перевірка та додавання звіту Box-Cox T-test
        if self.alpha_pred_boxcox is not None and self.distorted_data_backup is not None:
            alpha_true = getattr(self.app, 'true_alphas', [2.0, 0.5, 1.5]) 
            boxcox_report = alpha_t_test(
                original_data=self.app.stage1_data,
                distorted_data=self.distorted_data_backup,
                alpha_pred=self.alpha_pred_boxcox,
                alpha_true=alpha_true
            )
            rep_text.insert(tk.END, boxcox_report)
            rep_text.insert(tk.END, "="*65 + "\n\n")
        
        rep_text.insert(tk.END, f"Кількість точок (n): {n_points}\n\n")
        
        rep_text.insert(tk.END, "1. ГЛОБАЛЬНІ МЕТРИКИ ТОЧНОСТІ (Відносно ідеалу з Етапу 1)\n")
        rep_text.insert(tk.END, "-"*65 + "\n")
        rep_text.insert(tk.END, f"MSE (Квадратична похибка) : {mse:.4f}\n")
        rep_text.insert(tk.END, f"MAE (Абсолютна похибка)   : {mae:.4f}\n")
        rep_text.insert(tk.END, f"Max Error (Найгірша)      : {max_err:.4f}\n")
        rep_text.insert(tk.END, f"R² (Коеф. детермінації)   : {r2:.4f}\n\n")
        
        rep_text.insert(tk.END, "2. Т-ТЕСТ ПАРАМЕТРІВ РЕГРЕСІЇ\n")
        if hasattr(self.app, "true_coefs") and self.calculated_coefs is not None:
            true_coefs = np.array(self.app.true_coefs)
            pred_coefs = np.array(self.calculated_coefs)

            rep_text.insert(tk.END, proper_parameter_t_test(
                true_coefs, pred_coefs, X, Z_true, Z_pred
            ))
        
        rep_text.insert(tk.END, "ВИСНОВОК:\n")
        if p_val < 0.05:
            rep_text.insert(tk.END, "Різниця між ідеалом та знайденою площиною є статистично значущою (P-value < 0.05).\n")
            rep_text.insert(tk.END, "Це означає, що аналітичний метод (SVD) накопичив помітну похибку.")
        else:
            rep_text.insert(tk.END, "Статистично значущої різниці немає (P-value >= 0.05).\n")
            rep_text.insert(tk.END, "Аналітичний метод знайшов площину, що майже не відрізняється від ідеалу.")

        def copy_report():
            self.frame.clipboard_clear()
            self.frame.clipboard_append(rep_text.get(1.0, tk.END).strip())
            self.frame.update()
            messagebox.showinfo("Успіх", "Звіт скопійовано в буфер обміну!", parent=rep_win)

        ttk.Button(rep_win, text=" Скопіювати звіт", command=copy_report).pack(fill=tk.X, padx=10, pady=(0, 10))