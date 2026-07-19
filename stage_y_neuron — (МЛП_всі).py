import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple, Callable

# Імпортування функцій математичного ядра
from math_core import (
    calculate_svd_normal,
    get_plane_equation_coefs,
    alpha_t_test,
    proper_parameter_t_test
)

# Глобальні константи конфігурації навчання
DEFAULT_EPOCHS = "1000"
DEFAULT_BATCH_SIZE = 32
DEFAULT_FIXED_POINTS = 128
LOG_BOX_HEIGHT = 20
REPORT_WINDOW_SIZE = "750x650"

# Оптимізаційні гіперпараметри
LR_HYBRID = 0.005
LR_PURE = 0.002
ALPHA_SAFE_MIN = 0.5
ALPHA_SAFE_SCALE = 4.5
BOXCOX_EPSILON = 1e-8


# ============================================================
# АРХІТЕКТУРИ НЕЙРОМЕРЕЖ
# ============================================================

class CurvatureMLP(nn.Module):
    """Шукає коефіцієнти викривлення alpha на основі відсортованих даних за допомогою персептрона (MLP)."""
    
    def __init__(self, input_channels: int = 3, fixed_points: int = DEFAULT_FIXED_POINTS):
        super().__init__()
        self.fixed_points = fixed_points
        self.pool = nn.AdaptiveAvgPool1d(fixed_points)
        
        input_dim = input_channels * fixed_points
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pooled = self.pool(x)
        x_flat = x_pooled.reshape(x_pooled.size(0), -1) 
        alphas_raw = self.mlp(x_flat)
        alphas_safe = torch.sigmoid(alphas_raw) * ALPHA_SAFE_SCALE + ALPHA_SAFE_MIN
        return alphas_safe


class CoefRegressor(nn.Module):
    """MLP (в стилі PointNet), що шукає ТІЛЬКИ коефіцієнти площини a_i (БЕЗ вільного члена)."""
    
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
    """Чисто нейромережевий градієнтний граф: MLP (alpha) -> Restore -> MLP (coefs)."""
    
    def __init__(self, input_channels: int = 3, target_dim: int = 2):
        super().__init__()
        self.curvature_mlp = CurvatureMLP(input_channels)
        self.regressor = CoefRegressor(input_channels, target_dim)

    def forward(self, x_sorted: torch.Tensor, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alphas = self.curvature_mlp(x_sorted)
        alphas_reshaped = alphas.unsqueeze(2)
        
        x_restored = torch.sign(x_raw) * (torch.abs(x_raw) + 1e-6) ** (1.0 / alphas_reshaped)
        
        global_max = torch.max(torch.abs(x_restored), dim=2, keepdim=True)[0]
        global_max = torch.max(global_max, dim=1, keepdim=True)[0] + BOXCOX_EPSILON
        x_scaled = x_restored / global_max
        
        coefs = self.regressor(x_scaled)
        return alphas, coefs


# ============================================================
# ІНТЕРФЕЙС ТА ЛОГІКА 4-ГО ЕТАПУ
# ============================================================

class Stage4UI:
    """Управління навчанням нейромереж, прогнозуванням та генерацією аналітичних звітів етапу 4."""

    def __init__(self, frame: tk.Frame, app: Any):
        """Ініціалізує контекст обчислень та змінні стану моделей."""
        self.frame = frame
        self.app = app
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_training: bool = False
        self.epochs: int = 1000
        
        self.model_ns: Optional[nn.Module] = None  
        self.model_pn: Optional[nn.Module] = None  

        self.setup_ui()

    def setup_ui(self):
        """Конструює структуру елементів керування вікна."""
        ctrl_frame = ttk.LabelFrame(self.frame, text="Управління навчаннянням", padding=10)
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
        
        self.txt_log = tk.Text(log_frame, height=LOG_BOX_HEIGHT)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def log(self, message: str):
        """Виводить повідомлення в консоль інтерфейсу."""
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)

    def thread_safe_log(self, message: str):
        """Забезпечує безпечний виклик логування з фонових потоків."""
        self.frame.after(0, self.log, message)

    def _generate_batch(self, batch_size: int, n_points: int, n_features: int, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Генерує випадкові синтетичні гіперплощини для навчання моделей."""
        X = torch.rand(batch_size, target_dim, n_points, dtype=torch.float64, device=self.device) * 10.0
        a_true = torch.randn(batch_size, target_dim, dtype=torch.float64, device=self.device)
        
        Z = torch.zeros(batch_size, 1, n_points, dtype=torch.float64, device=self.device)
        for i in range(target_dim):
            Z[:, 0, :] += a_true[:, i:i+1] * X[:, i, :]
            
        data = torch.cat([X, Z], dim=1) 
        mins = data.min(dim=2, keepdim=True)[0]
        data = data - mins + 0.1
        
        alphas_true = torch.rand(batch_size, n_features, 1, dtype=torch.float64, device=self.device) * 4.0 + 0.5
        data_distorted = data ** alphas_true
        
        return data_distorted, alphas_true.squeeze(2), a_true

    def prepare_training(self) -> bool:
        """Перевіряє готовність системи до запуску циклу навчання."""
        if getattr(self.app, 'stage2_data', None) is None:
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
        """Запускає навчання гібридної нейро-символічної моделі у фоновому потоці."""
        if not self.prepare_training(): 
            return
        threading.Thread(target=self._training_loop, args=('neuro_symbolic',), daemon=True).start()

    def train_pure_neural(self):
        """Запускає навчання чистої нейромережевої моделі у фоновому потоці."""
        if not self.prepare_training(): 
            return
        threading.Thread(target=self._training_loop, args=('pure_neural',), daemon=True).start()

    def _training_loop(self, mode: str):
        """Уніфікований ітераційний цикл градієнтного навчання моделей."""
        self.thread_safe_log(f"=== Ініціалізація навчання [{mode.upper()}] на {self.device.type.upper()} ===")
        
        n_features = self.app.stage2_data.shape[1]
        target_dim = n_features - 1
        n_points = DEFAULT_FIXED_POINTS 
        batch_size = DEFAULT_BATCH_SIZE
        
        if mode == 'neuro_symbolic':
            model = CurvatureMLP(n_features, fixed_points=n_points).double().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=LR_HYBRID)
        else:
            model = PureNeuralModel(n_features, target_dim).double().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=LR_PURE)

        mse_loss_fn = nn.MSELoss()

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            x_distorted, alpha_true, coefs_true = self._generate_batch(batch_size, n_points, n_features, target_dim)
            x_sorted, _ = torch.sort(x_distorted, dim=2)
            d_min = x_sorted.min(dim=2, keepdim=True)[0]
            d_max = x_sorted.max(dim=2, keepdim=True)[0]
            x_norm = (x_sorted - d_min) / (d_max - d_min + BOXCOX_EPSILON)

            if mode == 'neuro_symbolic':
                alpha_pred = model(x_norm)
                loss = mse_loss_fn(alpha_pred, alpha_true)
            else:
                alpha_pred, a_pred = model(x_norm, x_distorted)
                loss = mse_loss_fn(alpha_pred, alpha_true) + mse_loss_fn(a_pred, coefs_true) 

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                loss_val = loss.item()
                alpha_err = torch.mean(torch.abs(alpha_pred - alpha_true)).item()
                
                self.app.training_history["epoch"].append(epoch)
                self.app.training_history["loss"].append(loss_val)
                self.app.training_history["alpha_error"].append(alpha_err)
                
                log_msg = f"Епоха {epoch:4d}/{self.epochs} | Loss: {loss_val:.4f} | Похибка α: {alpha_err:.4f}"
                if mode == 'pure_neural':
                    coef_err = torch.mean(torch.abs(a_pred - coefs_true)).item()
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

    def _get_plane_height_callback(self, coefs_key: str) -> Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]:
        """Генерує замикання функції розрахунку висоти площини регресії."""
        def plane_fn(x: np.ndarray, y: np.ndarray, idx_x: int, idx_y: int) -> np.ndarray:
            c = getattr(self.app, coefs_key)
            s = getattr(self.app, 'stage1_shifts', np.zeros(3))
            z = s[-1]
            if idx_x < len(c): 
                z += c[idx_x] * (x - s[idx_x])
            if idx_y < len(c): 
                z += c[idx_y] * (y - s[idx_y])
            return z
        return plane_fn

    def calculate_result(self):
        """Застосовує навчені нейромережі до користувацьких даних та формує 3D сцени."""
        if getattr(self.app, 'stage2_data', None) is None:
            return messagebox.showwarning("Увага", "Немає даних на Етапі 2.")
        if self.model_ns is None and self.model_pn is None:
            return messagebox.showwarning("Увага", "Спочатку проведіть навчання хоча б однієї моделі.")
            
        data = self.app.stage2_data
        n_features = data.shape[1]
        
        self.log("\n=== ЗАСТОСУВАННЯ МОДЕЛЕЙ ДО ВАШИХ ДАНИХ ===")
        
        x_tensor = torch.tensor(data, dtype=torch.float64, device=self.device).unsqueeze(0).transpose(1, 2)
        x_sorted, _ = torch.sort(x_tensor, dim=2)
        d_min = x_sorted.min(dim=2, keepdim=True)[0]
        d_max = x_sorted.max(dim=2, keepdim=True)[0]
        x_norm = (x_sorted - d_min) / (d_max - d_min + BOXCOX_EPSILON)

        custom_planes = {}
        restored_data_for_plot = None

        if self.model_ns is not None:
            self.log("\n--- Запуск Гібридної Мережі (MLP) ---")
            self.model_ns.eval()
            with torch.no_grad():
                alphas = self.model_ns(x_norm).cpu().numpy()[0]
                self.app.cnn_pred_alphas = alphas 
                self.log(f"Знайдено α: {np.round(alphas, 4)}")
                
                restored_ns = np.zeros_like(data)
                for i in range(n_features):
                    restored_ns[:, i] = np.sign(data[:, i]) * (np.abs(data[:, i]) + 1e-8) ** (1.0 / alphas[i])
                    
                normal, mean_vec = calculate_svd_normal(restored_ns)
                coefs_raw, _ = get_plane_equation_coefs(normal, mean_vec)
                self.app.stage4_ns_coefs = coefs_raw 
                self.log(f"Коефіцієнти a_i (SVD): {np.round(coefs_raw, 4)}")
                
                custom_planes["Гібрид (SVD)"] = (self._get_plane_height_callback('stage4_ns_coefs'), 'green')
                restored_data_for_plot = restored_ns

        if self.model_pn is not None:
            self.log("\n--- Запуск Pure Neural ---")
            self.model_pn.eval()
            with torch.no_grad():
                alphas, a_preds = self.model_pn(x_norm, x_tensor)
                self.app.stage4_pn_coefs = a_preds.cpu().numpy()[0]
                self.log(f"Знайдено α: {np.round(alphas.cpu().numpy()[0], 4)}")
                self.log(f"Коефіцієнти a_i (MLP): {np.round(self.app.stage4_pn_coefs, 4)}")
                
                custom_planes["Pure Neural (MLP)"] = (self._get_plane_height_callback('stage4_pn_coefs'), 'blue')
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

    def compare_methods(self):
        """Порівнює точність алгоритмів, розраховує статистичні метрики та будує звіт."""
        if not hasattr(self.app, 'stage4_ns_coefs') or not hasattr(self.app, 'stage4_pn_coefs'):
            return messagebox.showwarning("Увага", "Спочатку навчіть та застосуйте ОБИДВА методи (1, 2 і 3), щоб порівняти їх.")
            
        if getattr(self.app, 'stage1_data', None) is None:
            return messagebox.showwarning("Увага", "Для перевірки потрібні оригінальні лінійні дані (Етап 1).")
            
        data = self.app.stage1_data
        shifts = getattr(self.app, 'stage1_shifts', np.zeros(data.shape[1]))
        
        X = data[:, :-1]
        Z_true = data[:, -1]
        n_points = len(Z_true)
        
        def predict_z(coefs):
            pred = np.full_like(Z_true, shifts[-1])
            for i in range(len(coefs)):
                pred += coefs[i] * (X[:, i] - shifts[i])
            return pred
            
        Z_pred_ns = predict_z(self.app.stage4_ns_coefs)
        Z_pred_pn = predict_z(self.app.stage4_pn_coefs)
        
        res_ns, res_pn = Z_true - Z_pred_ns, Z_true - Z_pred_pn
        abs_res_ns, abs_res_pn = np.abs(res_ns), np.abs(res_pn)
        
        metrics = {
            "mse_ns": np.mean(res_ns**2), "mse_pn": np.mean(res_pn**2),
            "mae_ns": np.mean(abs_res_ns), "mae_pn": np.mean(abs_res_pn),
            "max_ns": np.max(abs_res_ns), "max_pn": np.max(abs_res_pn),
            "r2_ns": 1 - (np.sum(res_ns**2) / np.sum((Z_true - np.mean(Z_true))**2)),
            "r2_pn": 1 - (np.sum(res_pn**2) / np.sum((Z_true - np.mean(Z_true))**2))
        }
        
        t_stat, p_val = stats.ttest_rel(abs_res_ns, abs_res_pn)
        self._display_comparison_window(n_points, metrics, X, Z_true, Z_pred_ns, Z_pred_pn, t_stat, p_val)

    def _display_comparison_window(self, n_points: int, m: Dict[str, float], X: np.ndarray, 
                                   Z_true: np.ndarray, Z_pred_ns: np.ndarray, Z_pred_pn: np.ndarray, 
                                   t_stat: float, p_val: float):
        """Рендерить окреме діалогове вікно з порівняльним статистичним звітом етапу."""
        rep_win = tk.Toplevel(self.frame)
        rep_win.title("Звіт: Нейро-Символічний vs Pure Neural")
        rep_win.geometry(REPORT_WINDOW_SIZE)
        
        rep_text = tk.Text(rep_win, wrap=tk.WORD)
        rep_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        rep_text.insert(tk.END, "="*70 + "\n")
        rep_text.insert(tk.END, " ЗВІТ ПРО ПОРІВНЯННЯ АЛГОРИТМІВ НА ЕТАПІ 4\n")
        rep_text.insert(tk.END, "="*70 + "\n\n")
        
        # 1. Перевірка відновлення параметрів викривлення
        try:
            alpha_true = getattr(self.app, 'true_alphas', [2.0, 0.5, 1.5])
            mlp_alphas = getattr(self.app, 'cnn_pred_alphas', None)
            if mlp_alphas is not None:
                cnn_report = alpha_t_test(
                    original_data=self.app.stage1_data, distorted_data=self.app.stage2_data, 
                    alpha_pred=mlp_alphas, alpha_true=alpha_true
                )
                rep_text.insert(tk.END, "1. ПЕРЕВІРКА ГІПОТЕЗИ ДЛЯ MLP (ВІДНОВЛЕННЯ α)\n" + cnn_report + "="*70 + "\n\n")
        except Exception as e:
            rep_text.insert(tk.END, f"[Помилка обчислення T-тесту для MLP: {e}]\n\n")
        
        # 2. Таблиця порівняння параметрів моделі
        if hasattr(self.app, "true_coefs"):
            true_c, ns_c, pn_c = self.app.true_coefs, self.app.stage4_ns_coefs, self.app.stage4_pn_coefs
            rep_text.insert(tk.END, "2. ПОРІВНЯЛЬНИЙ АНАЛІЗ ПАРАМЕТРІВ ПЛОЩИНИ (a1, a2)\n" + "-"*70 + "\n")
            rep_text.insert(tk.END, f"{'Параметр':<12} | {'Ідеал':<15} | {'Гібрид (SVD)':<15} | {'Pure Neural':<15}\n" + "-"*70 + "\n")
            for i in range(len(true_c)):
                rep_text.insert(tk.END, f"a{i+1:<10} | {true_c[i]:<15.4f} | {ns_c[i]:<15.4f} | {pn_c[i]:<15.4f}\n")
            rep_text.insert(tk.END, "-"*70 + "\n\n")

        # 3. Метрики точності
        rep_text.insert(tk.END, "3. ГЛОБАЛЬНІ МЕТРИКИ ТОЧНОСТІ ТА Т-ТЕСТ ВІДХИЛЕНЬ\n" + "-"*70 + "\n")
        rep_text.insert(tk.END, f"Кількість точок (n): {n_points}\n\n")
        rep_text.insert(tk.END, f"{'Метрика':<20} | {'Нейро-Символ (SVD)':<22} | {'Pure Neural (MLP)':<20}\n" + "-"*70 + "\n")
        rep_text.insert(tk.END, f"{'MSE (Квадратична)':<20} | {m['mse_ns']:<22.4f} | {m['mse_pn']:<20.4f}\n")
        rep_text.insert(tk.END, f"{'MAE (Абсолютна)':<20} | {m['mae_ns']:<22.4f} | {m['mae_pn']:<20.4f}\n")
        rep_text.insert(tk.END, f"{'Max Error (Найгірша)':<20} | {m['max_ns']:<22.4f} | {m['max_pn']:<20.4f}\n")
        rep_text.insert(tk.END, f"{'R² (Пояснено)':<20} | {m['r2_ns']:<22.4f} | {m['r2_pn']:<20.4f}\n\n")
        
        # 4. Спеціалізований T-test коефіцієнтів площин
        if hasattr(self.app, "true_coefs"):
            true_c = np.array(self.app.true_coefs)
            rep_text.insert(tk.END, "\n=== ГІБРИДНИЙ МЕТОД (SVD) ===\n" + proper_parameter_t_test(true_c, self.app.stage4_ns_coefs, X, Z_true, Z_pred_ns))
            rep_text.insert(tk.END, "=== PURE NEURAL (MLP) ===\n" + proper_parameter_t_test(true_c, self.app.stage4_pn_coefs, X, Z_true, Z_pred_pn))

        # Висновок
        rep_text.insert(tk.END, "ВИСНОВОК:\n")
        if p_val < 0.05:
            better = "Нейро-Гібридний (SVD)" if m["mae_ns"] < m["mae_pn"] else "Нейромережевий (MLP)"
            rep_text.insert(tk.END, f"Різниця між методами є статистично значущою (P-value < 0.05, t={t_stat:.3f}).\nМетод '{better}' впорався об'єктивно краще.")
        else:
            rep_text.insert(tk.END, f"Статистично значущої різниці між методами немає (P-value >= 0.05, t={t_stat:.3f}).\nОбидві нейромережі відпрацювали однаково ефективно.")
        
        def _copy_callback():
            self.frame.clipboard_clear()
            self.frame.clipboard_append(rep_text.get(1.0, tk.END).strip())
            self.frame.update()
            messagebox.showinfo("Успіх", "Звіт скопійовано в буфер обміну!", parent=rep_win)

        ttk.Button(rep_win, text="Скопіювати звіт", command=_copy_callback).pack(fill=tk.X, padx=10, pady=5)