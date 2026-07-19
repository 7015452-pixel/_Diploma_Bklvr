import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Any, Dict, Tuple, Optional, Callable

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

# Підключаємо математичне ядро
from math_core import (
    calculate_svd_normal,
    get_plane_equation_coefs,
    alpha_t_test,
    proper_parameter_t_test
)

# ============================================================
# ГЛОБАЛЬНІ КОНСТАНТИ (КОНФІГУРАЦІЯ ТА ГІПЕРПАРАМЕТРИ)
# ============================================================
DEFAULT_EPOCHS: str = "1000"
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_POINTS: int = 128

# Гіперпараметри оптимізації
LR_NEURO_SYMBOLIC: float = 0.005
LR_PURE_NEURAL: float = 0.002

# Математичні константи стабілізації
EPSILON_STABILITY: float = 1e-8
BOX_COX_OFFSET: float = 1e-6

# Конфігурація генерації даних
ALPHA_MIN: float = 0.5
ALPHA_MAX: float = 4.5
DATA_SCALE_FACTOR: float = 10.0
DISTORTION_SHIFT: float = 0.1


# ============================================================
# АРХІТЕКТУРИ НЕЙРОМЕРЕЖ
# ============================================================

class CurvatureCNN(nn.Module):
    """Шукає коефіцієнти викривлення alpha на основі відсортованих даних[cite: 19]."""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2), 
            nn.ReLU(), 
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, input_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x).squeeze(-1)
        alphas_raw = self.fc(features)
        alphas_safe = torch.sigmoid(alphas_raw) * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN
        return alphas_safe


class CoefRegressor(nn.Module):
    """MLP (в стилі PointNet), що шукає ТІЛЬКИ коефіцієнти площини a_i (БЕЗ вільного члена)[cite: 19]."""
    
    def __init__(self, input_channels: int = 3, target_dim: int = 2):
        super().__init__()
        self.point_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1), 
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1), 
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1), 
            nn.ReLU()
        )
        self.global_fc = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, target_dim)
        )

    def forward(self, x_restored: torch.Tensor) -> torch.Tensor:
        features = self.point_conv(x_restored)
        global_features = torch.max(features, dim=2)[0] 
        coefs = self.global_fc(global_features)
        return coefs


class PureNeuralModel(nn.Module):
    """Чисто нейромережевий градієнтний граф: CNN (alpha) -> Restore -> MLP (coefs)[cite: 19]."""
    
    def __init__(self, input_channels: int = 3, target_dim: int = 2):
        super().__init__()
        self.cnn = CurvatureCNN(input_channels)
        self.regressor = CoefRegressor(input_channels, target_dim)

    def forward(self, x_sorted: torch.Tensor, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Знаходимо викривлення
        alphas = self.cnn(x_sorted)
        alphas_reshaped = alphas.unsqueeze(2)
        
        # 2. Відновлюємо простір
        x_restored = torch.sign(x_raw) * (torch.abs(x_raw) + BOX_COX_OFFSET) ** (1.0 / alphas_reshaped)
        
        # --- ФІКС МАСШТАБУ ДЛЯ ПЕРСЕПТРОНА ---
        global_max = torch.max(torch.abs(x_restored), dim=2, keepdim=True)[0]
        global_max = torch.max(global_max, dim=1, keepdim=True)[0] + EPSILON_STABILITY
        
        x_scaled = x_restored / global_max
        
        # 3. Прогнозування коефіцієнтів
        coefs = self.regressor(x_scaled)
        
        return alphas, coefs


# ============================================================
# ІНТЕРФЕЙС ТА ЛОГІКА 4-ГО ЕТАПУ
# ============================================================

class Stage4UI:
    """Клас керування інтерфейсом та бізнес-логікою обчислень 4-го етапу[cite: 19]."""

    def __init__(self, frame: tk.Frame, app: Any):
        self.frame = frame
        self.app = app
        
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_training: bool = False
        self.epochs: int = int(DEFAULT_EPOCHS)
        
        # Роздільне збереження архітектур моделей
        self.model_ns: Optional[nn.Module] = None  # Гібридна мережа (CurvatureCNN)
        self.model_pn: Optional[nn.Module] = None  # Чиста нейромережа (PureNeuralModel)

        self.setup_ui()

    def setup_ui(self):
        """Ініціалізація віджетів керування процесом навчання та відображення логів[cite: 19]."""
        ctrl_frame = ttk.LabelFrame(self.frame, text="Управління навчання", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="Кількість епох:").grid(row=0, column=0, sticky=tk.W)
        self.entry_epochs = ttk.Entry(ctrl_frame, width=10)
        self.entry_epochs.insert(0, DEFAULT_EPOCHS)
        self.entry_epochs.grid(row=0, column=1, padx=5)

        ttk.Button(ctrl_frame, text="1. Навчити (Гібридна)", command=self.train_neuro_symbolic).grid(row=1, column=0, pady=5, padx=5, sticky=tk.EW)
        ttk.Button(ctrl_frame, text="2. Навчити (Нейромережа)", command=self.train_pure_neural).grid(row=1, column=1, pady=5, padx=5, sticky=tk.EW)
        ttk.Button(ctrl_frame, text="3. Застосувати до даних та Візуалізувати", command=self.calculate_result).grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)
        ttk.Button(ctrl_frame, text="4. Порівняти методи (Звіт)", command=self.compare_methods).grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)

        log_frame = ttk.LabelFrame(self.frame, text="Консоль навчання", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.txt_log = tk.Text(log_frame, height=20)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def log(self, message: str):
        """Пряме додавання текстового рядка до вікна консолі[cite: 19]."""
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)

    def thread_safe_log(self, message: str):
        """Безпечний виклик логування з паралельних обчислювальних потоків[cite: 19]."""
        self.frame.after(0, self.log, message)

    def _generate_batch(self, batch_size: int, n_points: int, n_features: int, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Генерує випадкові гіперплощини без вільного члена для ітерації навчання[cite: 19]."""
        X = torch.rand(batch_size, target_dim, n_points, dtype=torch.float64, device=self.device) * DATA_SCALE_FACTOR
        a_true = torch.randn(batch_size, target_dim, dtype=torch.float64, device=self.device)
        
        Z = torch.zeros(batch_size, 1, n_points, dtype=torch.float64, device=self.device)
        for i in range(target_dim):
            Z[:, 0, :] += a_true[:, i:i+1] * X[:, i, :]
            
        data = torch.cat([X, Z], dim=1) 
        mins = data.min(dim=2, keepdim=True)[0]
        data = data - mins + DISTORTION_SHIFT
        
        alphas_true = torch.rand(batch_size, n_features, 1, dtype=torch.float64, device=self.device) * (ALPHA_MAX - ALPHA_MIN - 0.5) + ALPHA_MIN
        data_distorted = data ** alphas_true
        
        return data_distorted, alphas_true.squeeze(2), a_true

    def prepare_training(self) -> bool:
        """Валідація стану даних перед стартом тренувального циклу[cite: 19]."""
        if self.app.stage2_data is None:
            messagebox.showwarning("Увага", "Немає вхідних даних (Етап 2).")
            return False
        if self.is_training:
            messagebox.showinfo("Інфо", "Навчання вже йде!")
            return False
        try:
            self.epochs = int(self.entry_epochs.get())
        except ValueError:
            messagebox.showerror("Помилка", "Некоректне значення епох.")
            return False
            
        self.is_training = True
        self.txt_log.delete(1.0, tk.END)
        self.app.training_history = {
            "epoch": [], "loss": [], "alpha_error": [], "coef_error": [], "coefs": [], "true_coefs": []
        }
        return True

    def train_neuro_symbolic(self):
        """Ініціалізація асинхронного навчання гібридної архітектури[cite: 19]."""
        if not self.prepare_training(): 
            return
        threading.Thread(target=self._training_loop, args=('neuro_symbolic',), daemon=True).start()

    def train_pure_neural(self):
        """Ініціалізація асинхронного наскрізного навчання Pure Neural[cite: 19]."""
        if not self.prepare_training(): 
            return
        threading.Thread(target=self._training_loop, args=('pure_neural',), daemon=True).start()

    def _training_loop(self, mode: str):
        """Уніфікований та оптимізований цикл градієнтного спуску для обох моделей[cite: 19]."""
        self.thread_safe_log(f"=== Ініціалізація навчання [{mode.upper()}] на {self.device.type.upper()} ===")
        
        n_features: int = self.app.stage2_data.shape[1]
        target_dim: int = n_features - 1
        n_points: int = DEFAULT_NUM_POINTS
        batch_size: int = DEFAULT_BATCH_SIZE
        
        if mode == 'neuro_symbolic':
            model = CurvatureCNN(n_features).double().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=LR_NEURO_SYMBOLIC)
        else:
            model = PureNeuralModel(n_features, target_dim).double().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=LR_PURE_NEURAL)

        mse_loss_fn = nn.MSELoss()

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            x_distorted, alpha_true, coefs_true = self._generate_batch(batch_size, n_points, n_features, target_dim)
            x_sorted, _ = torch.sort(x_distorted, dim=2)
            d_min = x_sorted.min(dim=2, keepdim=True)[0]
            d_max = x_sorted.max(dim=2, keepdim=True)[0]
            x_norm = (x_sorted - d_min) / (d_max - d_min + EPSILON_STABILITY)

            if mode == 'neuro_symbolic':
                alpha_pred = model(x_norm)
                loss = mse_loss_fn(alpha_pred, alpha_true)
            else:
                alpha_pred, a_pred = model(x_norm, x_distorted)
                loss = mse_loss_fn(alpha_pred, alpha_true) + mse_loss_fn(a_pred, coefs_true) 

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                loss_val: float = loss.item()
                alpha_err: float = torch.mean(torch.abs(alpha_pred - alpha_true)).item()
                
                self.app.training_history["epoch"].append(epoch)
                self.app.training_history["loss"].append(loss_val)
                self.app.training_history["alpha_error"].append(alpha_err)
                
                log_msg = f"Епоха {epoch:4d}/{self.epochs} | Loss: {loss_val:.4f} | Похибка α: {alpha_err:.4f}"
                if mode == 'pure_neural':
                    coef_err: float = torch.mean(torch.abs(a_pred - coefs_true)).item()
                    self.app.training_history["coef_error"].append(coef_err)
                    log_msg += f" | Похибка a_i: {coef_err:.4f}"
                else:
                    self.app.training_history["coef_error"].append(0.0)

                self.thread_safe_log(log_msg)

        if mode == 'neuro_symbolic':
            self.model_ns = model
        else:
            self.model_pn = model

        self.thread_safe_log(f"\nНавчання {mode.upper()} завершено. Ви можете навчити іншу мережу або натиснути 'Застосувати'.")
        self.is_training = False

    def _get_plane_closure(self, coefs_attr: str) -> Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]:
        """Фабричний метод побудови функціональних замикань для генерації площин на графіку[cite: 19]."""
        def plane_function(x: np.ndarray, y: np.ndarray, idx_x: int, idx_y: int) -> np.ndarray:
            c = getattr(self.app, coefs_attr)
            s = getattr(self.app, 'stage1_shifts', np.zeros(3)) 
            z = s[-1] 
            if idx_x < len(c): 
                z += c[idx_x] * (x - s[idx_x])
            if idx_y < len(c): 
                z += c[idx_y] * (y - s[idx_y])
            return z
        return plane_function

    def calculate_result(self):
        """Обчислює параметри відновлення простору на основі реальних вхідних даних[cite: 19]."""
        if getattr(self.app, 'stage2_data', None) is None:
            return messagebox.showwarning("Увага", "Немає даних на Етапі 2.")
        if self.model_ns is None and self.model_pn is None:
            return messagebox.showwarning("Увага", "Спочатку проведіть навчання хоча б однієї моделі.")
            
        data: np.ndarray = self.app.stage2_data
        n_features: int = data.shape[1]
        
        self.log("\n=== ЗАСТОСУВАННЯ МОДЕЛЕЙ ДО ВАШИХ ДАНИХ ===")
        
        x_tensor = torch.tensor(data, dtype=torch.float64, device=self.device).unsqueeze(0).transpose(1, 2)
        x_sorted, _ = torch.sort(x_tensor, dim=2)
        d_min = x_sorted.min(dim=2, keepdim=True)[0]
        d_max = x_sorted.max(dim=2, keepdim=True)[0]
        x_norm = (x_sorted - d_min) / (d_max - d_min + EPSILON_STABILITY)

        custom_planes: Dict[str, Tuple[Callable, str]] = {}
        restored_data_for_plot: Optional[np.ndarray] = None

        # 1. Застосування Гібридної моделі (SVD)
        if self.model_ns is not None:
            self.log("\n--- Запуск Гібридної Мережі ---")
            self.model_ns.eval()
            with torch.no_grad():
                alphas = self.model_ns(x_norm).cpu().numpy()[0]
                self.app.cnn_pred_alphas = alphas 
                self.log(f"Знайдено α: {np.round(alphas, 4)}")
                
                restored_ns = np.zeros_like(data)
                for i in range(n_features):
                    restored_ns[:, i] = np.sign(data[:, i]) * (np.abs(data[:, i]) + BOX_COX_OFFSET) ** (1.0 / alphas[i])
                    
                normal, mean_vec = calculate_svd_normal(restored_ns)
                coefs_raw, _ = get_plane_equation_coefs(normal, mean_vec)
                self.app.stage4_ns_coefs = coefs_raw 
                self.log(f"Коефіцієнти a_i (SVD): {np.round(coefs_raw, 4)}")
                
                custom_planes["Гібрид (SVD)"] = (self._get_plane_closure('stage4_ns_coefs'), 'green')
                restored_data_for_plot = restored_ns

        # 2. Застосування наскрізної Pure Neural моделі (MLP)
        if self.model_pn is not None:
            self.log("\n--- Запуск Pure Neural ---")
            self.model_pn.eval()
            with torch.no_grad():
                alphas, a_preds = self.model_pn(x_norm, x_tensor)
                self.app.stage4_pn_coefs = a_preds.cpu().numpy()[0]
                self.log(f"Знайдено α: {np.round(alphas.cpu().numpy()[0], 4)}")
                self.log(f"Коефіцієнти a_i (MLP): {np.round(self.app.stage4_pn_coefs, 4)}")
                
                custom_planes["Pure Neural (MLP)"] = (self._get_plane_closure('stage4_pn_coefs'), 'blue')
                if restored_data_for_plot is None:
                    restored_data_for_plot = data 

        self.app.stage4_restored_data = restored_data_for_plot
        
        self.app.show_plot(
            4,
            {"Дані для аналізу": restored_data_for_plot},
            "Етап 4: Відновлені площини",
            show_plane=True, 
            custom_planes=custom_planes
        )

    def _calculate_regression_metrics(self, Z_true: np.ndarray, Z_pred: np.ndarray) -> Dict[str, Any]:
        """Декомпонований метод розрахунку аналітичних регресійних метрик[cite: 19]."""
        residuals = Z_true - Z_pred
        abs_residuals = np.abs(residuals)
        
        mse = np.mean(residuals ** 2)
        mae = np.mean(abs_residuals)
        max_err = np.max(abs_residuals)
        r2 = 1 - (np.sum(residuals ** 2) / np.sum((Z_true - np.mean(Z_true)) ** 2))
        
        return {
            "residuals": residuals,
            "abs_residuals": abs_residuals,
            "mse": mse,
            "mae": mae,
            "max": max_err,
            "r2": r2
        }

    def _execute_clipboard_copy(self, text_widget: tk.Text, root_window: tk.Toplevel):
        """Потокобезпечне та явне копіювання звіту до системного буфера обміну[cite: 19]."""
        self.frame.clipboard_clear()
        self.frame.clipboard_append(text_widget.get(1.0, tk.END).strip())
        self.frame.update()  # Гарантує оновлення контексту вікна в ОС
        messagebox.showinfo("Успіх", "Звіт успішно скопійовано в буфер обміну!", parent=root_window)

    def compare_methods(self):
        """Виконує повний порівняльний та статистичний аналіз відновлених поверхонь[cite: 19]."""
        if not hasattr(self.app, 'stage4_ns_coefs') or not hasattr(self.app, 'stage4_pn_coefs'):
            return messagebox.showwarning("Увага", "Спочатку навчіть та застосуйте ОБИДВА методи (1, 2 і 3).")
            
        if getattr(self.app, 'stage1_data', None) is None:
            return messagebox.showwarning("Увага", "Для перевірки потрібні оригінальні лінійні дані (Етап 1).")
            
        data: np.ndarray = self.app.stage1_data
        shifts: np.ndarray = getattr(self.app, 'stage1_shifts', np.zeros(data.shape[1]))
        
        X: np.ndarray = data[:, :-1]
        Z_true: np.ndarray = data[:, -1]
        n_points: int = len(Z_true)
        
        def predict_z(coefs: np.ndarray) -> np.ndarray:
            z_pred = np.full_like(Z_true, shifts[-1])
            for i in range(len(coefs)):
                z_pred += coefs[i] * (X[:, i] - shifts[i])
            return z_pred
            
        Z_pred_ns = predict_z(self.app.stage4_ns_coefs)
        Z_pred_pn = predict_z(self.app.stage4_pn_coefs)
        
        metrics_ns = self._calculate_regression_metrics(Z_true, Z_pred_ns)
        metrics_pn = self._calculate_regression_metrics(Z_true, Z_pred_pn)
        
        t_stat, p_val = stats.ttest_rel(metrics_ns["abs_residuals"], metrics_pn["abs_residuals"])
        
        # Створення UI вікна звіту
        rep_win = tk.Toplevel(self.frame)
        rep_win.title("Звіт: Нейро-Символічний vs Pure Neural")
        rep_win.geometry("750x650")
        
        rep_text = tk.Text(rep_win, wrap=tk.WORD)
        rep_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        rep_text.insert(tk.END, "="*70 + "\n")
        rep_text.insert(tk.END, " ЗВІТ ПРО ПОРІВНЯННЯ АЛГОРИТМІВ НА ЕТАПІ 4\n")
        rep_text.insert(tk.END, "="*70 + "\n\n")
        
        # 1. Перевірка Альфи для CNN
        try:
            alpha_true = getattr(self.app, 'true_alphas', [2.0, 0.5, 1.5])
            cnn_alphas = getattr(self.app, 'cnn_pred_alphas', None)
            
            if cnn_alphas is not None:
                cnn_report = alpha_t_test(
                    original_data=self.app.stage1_data,
                    distorted_data=self.app.stage2_data, 
                    alpha_pred=cnn_alphas,
                    alpha_true=alpha_true
                )
                rep_text.insert(tk.END, "1. ПЕРЕВІРКА ГІПОТЕЗИ ДЛЯ CNN (ВІДНОВЛЕННЯ α)\n")
                rep_text.insert(tk.END, cnn_report)
                rep_text.insert(tk.END, "="*70 + "\n\n")
        except Exception as e:
            rep_text.insert(tk.END, f"[Помилка обчислення T-тесту для CNN: {e}]\n\n")
        
        # 2. Порівняльна таблиця коефіцієнтів
        if hasattr(self.app, "true_coefs"):
            true_c = self.app.true_coefs
            ns_c = self.app.stage4_ns_coefs
            pn_c = self.app.stage4_pn_coefs
            
            rep_text.insert(tk.END, "2. ПОРІВНЯЛЬНИЙ АНАЛІЗ ПАРАМЕТРІВ ПЛОЩИНИ (a1, a2)\n")
            rep_text.insert(tk.END, "-"*70 + "\n")
            rep_text.insert(tk.END, f"{'Параметр':<12} | {'Ідеал':<15} | {'Гібрид (SVD)':<15} | {'Pure Neural':<15}\n")
            rep_text.insert(tk.END, "-"*70 + "\n")
            
            for i in range(len(true_c)):
                rep_text.insert(tk.END, f"a{i+1:<10} | {true_c[i]:<15.4f} | {ns_c[i]:<15.4f} | {pn_c[i]:<15.4f}\n")
            rep_text.insert(tk.END, "-"*70 + "\n\n")

        # 3. Метрики точності
        rep_text.insert(tk.END, "3. ГЛОБАЛЬНІ МЕТРИКИ ТОЧНОСТІ ТА Т-ТЕСТ ВІДХИЛЕНЬ\n")
        rep_text.insert(tk.END, "-"*70 + "\n")
        rep_text.insert(tk.END, f"Кількість точок (n): {n_points}\n\n")
        rep_text.insert(tk.END, f"{'Метрика':<20} | {'Нейро-Символ (SVD)':<22} | {'Pure Neural (MLP)':<20}\n")
        rep_text.insert(tk.END, "-"*70 + "\n")
        rep_text.insert(tk.END, f"{'MSE (Квадратична)':<20} | {metrics_ns['mse']:<22.4f} | {metrics_pn['mse']:<20.4f}\n")
        rep_text.insert(tk.END, f"{'MAE (Абсолютна)':<20} | {metrics_ns['mae']:<22.4f} | {metrics_pn['mae']:<20.4f}\n")
        rep_text.insert(tk.END, f"{'Max Error (Найгірша)':<20} | {metrics_ns['max']:<22.4f} | {metrics_pn['max']:<20.4f}\n")
        rep_text.insert(tk.END, f"{'R² (Пояснено)':<20} | {metrics_ns['r2']:<22.4f} | {metrics_pn['r2']:<20.4f}\n\n")
        
        # 4. Параметричний T-тест
        if hasattr(self.app, "true_coefs"):
            true_c_arr = np.array(self.app.true_coefs)
            
            rep_text.insert(tk.END, "\n=== ГІБРИДНИЙ МЕТОД (SVD) ===\n")
            rep_text.insert(tk.END, proper_parameter_t_test(
                true_c_arr, self.app.stage4_ns_coefs, X, Z_true, Z_pred_ns
            ))
            
            rep_text.insert(tk.END, "=== PURE NEURAL (MLP) ===\n")
            rep_text.insert(tk.END, proper_parameter_t_test(
                true_c_arr, self.app.stage4_pn_coefs, X, Z_true, Z_pred_pn
            ))

        rep_text.insert(tk.END, "ВИСНОВОК:\n")
        if p_val < 0.05:
            better = "Нейро-Гібридний (SVD)" if metrics_ns["mae"] < metrics_pn["mae"] else "Нейромережевий (MLP)"
            rep_text.insert(tk.END, f"Різниця між методами є статистично значущою (P-value < 0.05, t={t_stat:.3f}).\n")
            rep_text.insert(tk.END, f"Метод '{better}' впорався об'єктивно краще.")
        else:
            rep_text.insert(tk.END, f"Статистично значущої різниці між методами немає (P-value >= 0.05, t={t_stat:.3f}).\n")
            rep_text.insert(tk.END, f"Обидві нейромережі відпрацювали однаково ефективно.")
        
        # Виклик безпечного методу через єдину команду без об'єднання в масив lambda-виразу
        ttk.Button(
            rep_win,
            text="Скопіювати звіт",
            command=lambda: self._execute_clipboard_copy(rep_text, rep_win)
        ).pack(fill=tk.X, padx=10, pady=5)