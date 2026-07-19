import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import Any, Dict, List, Tuple, Optional, Callable
import pandas as pd
import numpy as np
import scipy.stats as stats
import threading

import torch
import torch.nn as nn

# ============================================================
# ГЛОБАЛЬНІ КОНСТАНТИ (КОНФІГУРАЦІЯ ТА ГІПЕРПАРАМЕТРИ)
# ============================================================
# Математичні константи та відступи стабілізації
EPSILON_STABILITY: float = 1e-8
BOX_COX_MIN_OFFSET: float = 1e-5
BOX_COX_NEURAL_OFFSET: float = 1e-6

# Фільтрація аномальних значень
SIGMA_THRESHOLD: float = 3.0

# Константи масштабування та відображення UI
TEXT_BOX_HEIGHT_LARGE: int = 18
TEXT_BOX_HEIGHT_SMALL: int = 16
COMBOBOX_DEFAULT_WIDTH: int = 45
TREEVIEW_DEFAULT_WIDTH: int = 80

# Межі зміни коефіцієнтів трансформації
ALPHA_MIN: float = 0.5
ALPHA_MAX: float = 5.0  # 4.5 + 0.5 відповідно до початкової логіки сигмоїди


# ============================================================
# АРХІТЕКТУРИ МЕРЕЖ
# ============================================================

class CurvatureCNN(nn.Module):
    """Шукає коефіцієнти викривлення alpha на основі відсортованих даних[cite: 20]."""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, input_channels) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x).squeeze(-1)
        alphas_raw = self.fc(features)
        alphas_safe = torch.sigmoid(alphas_raw) * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN 
        return alphas_safe


class CoefRegressor(nn.Module):
    """MLP (в стилі PointNet), що шукає ТІЛЬКИ коефіцієнти площини a_i[cite: 20]."""
    
    def __init__(self, input_channels: int = 3, target_dim: int = 2):
        super().__init__()
        self.point_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1), nn.ReLU()
        )
        self.global_fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, target_dim) 
        )

    def forward(self, x_restored: torch.Tensor) -> torch.Tensor:
        features = self.point_conv(x_restored)
        global_features = torch.max(features, dim=2)[0] 
        coefs = self.global_fc(global_features)
        return coefs


class PureNeuralModel(nn.Module):
    """Чисто нейромережевий градієнтний граф: CNN (alpha) -> Restore -> MLP (coefs)[cite: 20]."""
    
    def __init__(self, input_channels: int = 3, target_dim: int = 2):
        super().__init__()
        self.cnn = CurvatureCNN(input_channels)
        self.regressor = CoefRegressor(input_channels, target_dim)

    def forward(self, x_sorted: torch.Tensor, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alphas = self.cnn(x_sorted)
        alphas_reshaped = alphas.unsqueeze(2)
        
        x_restored = torch.sign(x_raw) * (torch.abs(x_raw) + BOX_COX_NEURAL_OFFSET) ** (1.0 / alphas_reshaped)
        
        global_max = torch.max(torch.abs(x_restored), dim=2, keepdim=True)[0]
        global_max = torch.max(global_max, dim=1, keepdim=True)[0] + EPSILON_STABILITY
        
        x_scaled = x_restored / global_max
        coefs = self.regressor(x_scaled)
        
        return alphas, coefs


# ============================================================
# ІНТЕРФЕЙС ТА ЛОГІКА ВКАДКИ ФАЙЛІВ
# ============================================================

class FileTabUI:
    """Уніфікований UI-модуль завантаження, первинної обробки та аналізу даних[cite: 20]."""

    def __init__(self, parent_frame: tk.Frame, app_controller: Any):
        self.app = app_controller
        self.frame = parent_frame
        
        # Контекст збереження даних таблиць
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None  
        
        # Прапорці стану поточної обробки
        self.is_shifted: bool = False
        self.is_normalized: bool = False
        
        # Результати розрахунків та параметри площин
        self.analyt_coefs: Optional[np.ndarray] = None
        self.analyt_mean_vec: Optional[np.ndarray] = None
        self.neural_coefs: Optional[np.ndarray] = None
        self.neural_mean_vec: Optional[np.ndarray] = None
        self.active_neural_mode: Optional[str] = None
        
        # Буферизовані проміжні масиви трансформацій
        self.bc_data_full: Optional[np.ndarray] = None
        self.bc_data_filtered: Optional[np.ndarray] = None
        self.neural_data_full: Optional[np.ndarray] = None
        self.neural_data_filtered: Optional[np.ndarray] = None

        self.setup_ui()

    def setup_ui(self):
        """Ініціалізація та пакування всіх елементів керування вікна[cite: 20]."""
        top_container = ttk.Frame(self.frame)
        top_container.pack(fill=tk.X, pady=5, padx=5)

        # Рядок дій файлової системи та очищення
        row1 = ttk.Frame(top_container)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Відкрити файл (TXT/CSV)", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Видалити рядки з '0'/NaN", command=self.clean_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Зберегти зміни у файл", command=self.save_file).pack(side=tk.LEFT, padx=2)

        # Рядок лінійних математичних трансформацій
        row2 = ttk.Frame(top_container)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text=" Скинути дані", command=self.reset_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text=" Зсув до додатніх", command=self.shift_positive).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text=" Нормування (MaxAbs)", command=self.normalize_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text=" Стандартизація (Z-score)", command=self.standardize_data).pack(side=tk.LEFT, padx=2)

        # Середня роздільна панель (Таблиця + Конфігурація)
        self.paned_mid = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self.paned_mid.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        table_frame = ttk.Frame(self.paned_mid)
        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        self.tree = ttk.Treeview(table_frame, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_double_click)
        
        self.paned_mid.add(table_frame, weight=3)

        # Конфігурація вибору цільових змінних моделей
        col_frame = ttk.LabelFrame(self.paned_mid, text=" Конфігурація моделі:", padding=5)
        
        ttk.Label(col_frame, text="Залежна змінна (Z):", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(2, 2))
        self.cb_dep_var = ttk.Combobox(col_frame, state="readonly")
        self.cb_dep_var.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(col_frame, text="Незалежні змінні (X):").pack(anchor=tk.W, pady=(0, 2))
        list_container = ttk.Frame(col_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        list_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL)
        self.listbox_cols = tk.Listbox(list_container, selectmode=tk.MULTIPLE, yscrollcommand=list_scroll.set)
        list_scroll.config(command=self.listbox_cols.yview)
        
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_cols.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.paned_mid.add(col_frame, weight=1)

        # Нижня частина вікна для виведення результатів та логів текстових боксів
        bottom_frame = ttk.Frame(self.frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        columns_container = ttk.Frame(bottom_frame)
        columns_container.pack(fill=tk.BOTH, expand=True, pady=2)

        col1 = ttk.Frame(columns_container)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        ttk.Button(col1, text="1. Розрахувати статистику", command=self.calc_statistics).pack(fill=tk.X, pady=2)
        self.stat_box = tk.Text(col1, height=TEXT_BOX_HEIGHT_LARGE, wrap=tk.WORD)
        self.stat_box.pack(fill=tk.BOTH, expand=True)

        col2 = ttk.Frame(columns_container)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        ttk.Button(col2, text="2. Аналітичний розв'язок", command=self.run_analytical).pack(fill=tk.X, pady=2)
        self.analytical_box = tk.Text(col2, height=TEXT_BOX_HEIGHT_LARGE, wrap=tk.WORD)
        self.analytical_box.pack(fill=tk.BOTH, expand=True)

        col3 = ttk.Frame(columns_container)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        ttk.Button(col3, text="3. Гібридний метод (.pt)", command=lambda: self.run_neural('ns')).pack(fill=tk.X, pady=2)
        ttk.Button(col3, text="3. Нейромережевий метод (.pt)", command=lambda: self.run_neural('pn')).pack(fill=tk.X, pady=2)
        self.neural_box = tk.Text(col3, height=TEXT_BOX_HEIGHT_SMALL, wrap=tk.WORD)
        self.neural_box.pack(fill=tk.BOTH, expand=True)

        # Блок візуалізації та побудови графіків
        vis_frame = ttk.Frame(bottom_frame)
        vis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(vis_frame, text="Дані для графіка та площин:").pack(side=tk.LEFT, padx=5)
        self.plot_state_cb = ttk.Combobox(vis_frame, state="readonly", values=[
            "1. До трансформації (Оригінал)",
            "2. Після трансформації (Box-Cox / Відновлені)",
            "3. З вилученням аномалій за правилом 3-сігма"
        ], width=COMBOBOX_DEFAULT_WIDTH)
        self.plot_state_cb.current(2)  
        self.plot_state_cb.pack(side=tk.LEFT, padx=5)

        ttk.Button(vis_frame, text=" 4. Візуальне порівняння площин", command=self.vis_plane_comparison).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Панель фінальних звітів
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text=" 5. Порівняти методи (T-test Стьюдента)", command=self.compare_methods).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text=" Скопіювати усі результати", command=self.copy_all_results).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    
    def load_file(self):
        """Здійснює вибір файлу на диску та парсинг текстової структури ознак[cite: 20]."""
        filepath = filedialog.askopenfilename(filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")])
        if not filepath: 
            return
        try:
            self.df = pd.read_csv(filepath, sep=r'\s+', header=0, engine='python')
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '.'), errors='coerce')

            self.original_df = self.df.copy()
            self.is_shifted = False
            self.is_normalized = False
            self.analyt_coefs = None
            self.neural_coefs = None
            
            self.update_listbox()
            self.update_treeview()
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося прочитати файл:\n{e}")

    def clean_data(self):
        """Очищує нульові та відсутні значення, обчислюючи матриці кореляцій[cite: 20]."""
        if self.df is None: 
            return
        corr_before = self.df.corr(method='pearson')
        
        self.df = self.df.replace([0, 0.0, '0', '0.0'], np.nan)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.original_df = self.df.copy()
        
        self.update_listbox()
        self.update_treeview()

        corr_after = self.df.corr(method='pearson')
        
        selected_indices = self.listbox_cols.curselection()
        selected_cols = [self.listbox_cols.get(i) for i in selected_indices]
        if self.cb_dep_var.get() and self.cb_dep_var.get() not in selected_cols:
            selected_cols.append(self.cb_dep_var.get())
            
        kendall_info = ""
        if len(selected_cols) > 1:
            kendall_corr = self.df[selected_cols].corr(method='kendall')
            kendall_info = f"\nКореляція Кенделла (для обраних векторів):\n{kendall_corr.to_string()}\n"
            
        messagebox.showinfo("Очищення", f"Рядки з 0 або NaN видалено.\n\nЛінійна кореляція (Pearson) ДО:\n{corr_before.to_string()}\n\nЛінійна кореляція (Pearson) ПІСЛЯ:\n{corr_after.to_string()}\n{kendall_info}")

    def reset_data(self):
        """Повертає структуру таблиці до вихідного немодифікованого стану[cite: 20]."""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.is_shifted = False
            self.is_normalized = False
            self.analyt_coefs = None
            self.neural_coefs = None
            self.update_treeview()

    def shift_positive(self):
        """Здійснює трансляційний позитивний зсув по осях значень ознак[cite: 20]."""
        if self.df is None: 
            return
        for col in self.df.columns:
            min_val = self.df[col].min()
            if min_val <= 0:
                self.df[col] += (np.abs(min_val) + 0.01)
        self.is_shifted = True
        self.update_treeview()

    def normalize_data(self):
        """Виконує нормалізацію відносного максимуму модуля (MaxAbs)[cite: 20]."""
        if self.df is None: 
            return
        for col in self.df.columns:
            max_abs = self.df[col].abs().max()
            if max_abs > 0:
                self.df[col] /= max_abs
        self.is_normalized = True
        self.update_treeview()

    def save_file(self):
        """Зберігає модифікований локальний фрейм у форматі CSV[cite: 20]."""
        if self.df is None: 
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filepath:
            self.df.to_csv(filepath, index=False)

    def standardize_data(self):
        """Трансформує вектори значень методом Z-score стандартизації[cite: 20]."""
        if self.df is None: 
            return
        means = self.df.mean()
        stds = self.df.std()
        self.df = (self.df - means) / (stds + EPSILON_STABILITY)
        self.is_normalized = True
        self.update_treeview()
        
        info = "Стандартизацію (Z-score) застосовано. Параметри:\n\n"
        for col in self.df.columns:
            info += f"{col} -> Середнє: {means[col]:.4f}, Відх.: {stds[col]:.4f}\n"
        messagebox.showinfo("Стандартизація", info)

    def on_double_click(self, event: tk.Event):
        """Забезпечує можливість ручного редагування комірок таблиці по подвійному кліку[cite: 20]."""
        if self.df is None: 
            return
        try:
            item = self.tree.selection()[0]
            col_idx = int(self.tree.identify_column(event.x).replace('#', '')) - 1
            row_idx = self.tree.index(item)
            old_val = self.df.iat[row_idx, col_idx]
            new_val = simpledialog.askstring("Редагування", "Введіть нове значення:", initialvalue=str(old_val))
            if new_val is not None:
                self.df.iat[row_idx, col_idx] = float(new_val.replace(',', '.'))
                self.update_treeview()
        except Exception:
            pass

    def update_treeview(self):
        """Синхронізує поточний Pandas DataFrame з графічним віджетом Treeview[cite: 20]."""
        self.tree.delete(*self.tree.get_children())
        if self.df is None: 
            return
        self.tree["column"] = list(self.df.columns)
        self.tree["show"] = "headings"
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=TREEVIEW_DEFAULT_WIDTH, anchor=tk.CENTER)
        for row in self.df.to_numpy().tolist():
            self.tree.insert("", "end", values=row)

    def update_listbox(self):
        """Оновлює списки вибору колонок ознак відповідно до структури даних[cite: 20]."""
        self.listbox_cols.delete(0, tk.END)
        cols = list(self.df.columns)
        for col in cols: 
            self.listbox_cols.insert(tk.END, col)
        self.cb_dep_var['values'] = cols
        if cols: 
            self.cb_dep_var.current(len(cols) - 1)

    def get_selected_data(self) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
        """Повертає структуровані вектори підмножини обраних змінних[cite: 20]."""
        if self.df is None: 
            return None, None
        selected_indices = self.listbox_cols.curselection()
        if not selected_indices: 
            return None, None
        dep_col = self.cb_dep_var.get()
        indep_cols = [self.listbox_cols.get(i) for i in selected_indices]
        if dep_col in indep_cols: 
            indep_cols.remove(dep_col)
        working_cols = indep_cols + [dep_col]
        return working_cols, self.df[working_cols].to_numpy()

    def apply_mnk(self, data: np.ndarray, cols: List[str], text_box: tk.Text) -> np.ndarray:
        """Обчислює коефіцієнти площини методом найменших квадратів (МНК)[cite: 20]."""
        text_box.insert(tk.END, "Знайдено параметри:\n")
        
        X = data[:, :-1]
        Z = data[:, -1]
        
        mean_X = np.mean(X, axis=0)
        mean_Z = np.mean(Z)
        
        X_centered = X - mean_X
        Z_centered = Z - mean_Z
        
        a_coefs, _, _, _ = np.linalg.lstsq(X_centered, Z_centered, rcond=None)
        
        for i in range(len(a_coefs)):
            text_box.insert(tk.END, f"  a{i+1} ({cols[i]}): {a_coefs[i]:.6f}\n")
            
        text_box.see(tk.END)
        return a_coefs

    def calc_statistics(self):
        """Аналізує вищі статистичні моменти розподілу та рангову кореляцію[cite: 20]."""
        if self.df is None: 
            return messagebox.showwarning("Увага", "Завантажте файл!")
        cols, _ = self.get_selected_data()
        if cols is None: 
            cols = list(self.df.columns)

        self.stat_box.delete(1.0, tk.END)
        self.stat_box.insert(tk.END, "=== СТАТИСТИКА ТА МЕТРИКИ ===\n")
        
        for col in cols:
            series = self.df[col].dropna()
            self.stat_box.insert(tk.END, f"Ознака [{col}]:\n")
            self.stat_box.insert(tk.END, f"  Мінімум: {series.min():.4f}\n")
            self.stat_box.insert(tk.END, f"  Максимум: {series.max():.4f}\n")
            self.stat_box.insert(tk.END, f"  Середнє: {series.mean():.4f}\n")
            self.stat_box.insert(tk.END, f"  Стандартне відхилення: {series.std():.4f}\n")
            self.stat_box.insert(tk.END, f"  Асиметрія (Skewness): {stats.skew(series):.4f}\n")
            self.stat_box.insert(tk.END, f"  Ексцес (Kurtosis): {stats.kurtosis(series):.4f}\n\n")

        if len(cols) > 1:
            self.stat_box.insert(tk.END, "\n=== КОРЕЛЯЦІЙНИЙ АНАЛІЗ ===\n")
            corr_matrix = self.df[cols].corr(method='kendall')
            
            self.stat_box.insert(tk.END, "Матриця рангової кореляції Кендалла:\n")
            header_str = f"{'Ознака':<12}" + "".join([f"{c:>10}" for c in cols]) + "\n"
            self.stat_box.insert(tk.END, header_str)
            self.stat_box.insert(tk.END, "-" * (12 + len(cols) * 10) + "\n")
            
            for row_col in cols:
                row_str = f"{row_col:<12}"
                for col_col in cols:
                    row_str += f"{corr_matrix.loc[row_col, col_col]:>10.4f}"
                self.stat_box.insert(tk.END, row_str + "\n")
        self.stat_box.see(tk.END)

    def _print_correlation_matrix(self, text_box: tk.Text, corr_df: pd.DataFrame, cols: List[str]):
        """Допоміжний декомпонований метод виведення матриць лінійної залежності[cite: 20]."""
        header_str = f"{'Ознака':<12}" + "".join([f"{c:>10}" for c in cols]) + "\n"
        text_box.insert(tk.END, header_str)
        text_box.insert(tk.END, "-" * (12 + len(cols) * 10) + "\n")
        for row_col in cols:
            row_str = f"{row_col:<12}"
            for col_col in cols:
                row_str += f"{corr_df.loc[row_col, col_col]:>10.4f}"
            text_box.insert(tk.END, row_str + "\n")

    def run_analytical(self):
        """Аналітичний контур: нормалізація Box-Cox, фільтрація сигма-викидів та МНК[cite: 20]."""
        cols, data = self.get_selected_data()
        if cols is None: 
            return
        self.analytical_box.delete(1.0, tk.END)
        self.analytical_box.insert(tk.END, "=== 1. НОРМАЛІЗАЦІЯ (Box-Cox) ===\n")
        
        num_dimensions = data.shape[1]
        bc_data = np.zeros_like(data)
        lambdas: List[float] = [] 
        
        for i in range(num_dimensions):
            col_data = data[:, i].copy()
            if np.any(col_data <= 0):
                col_data = col_data - np.min(col_data) + BOX_COX_MIN_OFFSET
            lmb = stats.boxcox_normmax(col_data)
            lambdas.append(lmb) 
            bc_data[:, i] = stats.boxcox(col_data, lmbda=lmb)
            
        self.analytical_box.insert(tk.END, "=== ПАРАМЕТРИ ПЕРЕТВОРЕНЬ ===\n")
        if lambdas:
            self.analytical_box.insert(tk.END, f"Параметри нормалізації (Box-Cox λ): {np.round(lambdas, 4)}\n")
        self.bc_data_full = bc_data  
        
        corr_before = pd.DataFrame(bc_data, columns=cols).corr(method='pearson')
        self.analytical_box.insert(tk.END, "=== КОРЕЛЯЦІЙНИЙ АНАЛІЗ ===\n")
        self.analytical_box.insert(tk.END, "Матриця лінійної кореляції Пірсона (до фільтрації):\n")
        self._print_correlation_matrix(self.analytical_box, corr_before, cols)
            
        self.analytical_box.insert(tk.END, f"\n=== 2. ФІЛЬТРАЦІЯ ({int(SIGMA_THRESHOLD)}-СІГМА) ===\n")
        z_scores = np.abs(stats.zscore(bc_data))
        mask = (z_scores < SIGMA_THRESHOLD).all(axis=1)
        self.bc_data_filtered = bc_data[mask]
        self.analytical_box.insert(tk.END, f"Залишено точок: {len(self.bc_data_filtered)} з {len(bc_data)}\n\n")
        
        corr_after = pd.DataFrame(self.bc_data_filtered, columns=cols).corr(method='pearson')
        self.analytical_box.insert(tk.END, "Матриця лінійної кореляції Пірсона (ПІСЛЯ фільтрації):\n")
        self._print_correlation_matrix(self.analytical_box, corr_after, cols)
            
        self.analyt_coefs = self.apply_mnk(self.bc_data_filtered, cols, self.analytical_box)
        self.analyt_mean_vec = np.mean(self.bc_data_filtered, axis=0)

    def run_neural(self, mode: str):
        """Запуск фонового асинхронного розрахунку нейромережевих моделей[cite: 20]."""
        cols, data = self.get_selected_data()
        if cols is None: 
            return
        
        filepath = filedialog.askopenfilename(title="Оберіть навчену модель (.pt)", filetypes=[("PyTorch Model", "*.pt")])
        if not filepath: 
            return
        
        self.neural_box.delete(1.0, tk.END)
        method_name = "Гібридної" if mode == 'ns' else "Нейромережевої"
        self.neural_box.insert(tk.END, f"Завантаження {method_name} моделі...\nБудь ласка, зачекайте...\n")
        self.frame.update()
        self.active_neural_mode = mode

        def _pytorch_worker():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                n_cols = data.shape[1]
                target_dim = n_cols - 1
                
                if mode == 'ns':
                    model = CurvatureCNN(input_channels=n_cols).double().to(device)
                else:
                    model = PureNeuralModel(input_channels=n_cols, target_dim=target_dim).double().to(device)
                    
                model.load_state_dict(torch.load(filepath, map_location=device))
                model.eval()
                
                distorted_sorted = np.sort(data, axis=0)
                d_min = np.min(distorted_sorted, axis=0, keepdims=True)
                d_max = np.max(distorted_sorted, axis=0, keepdims=True)
                distorted_normalized = (distorted_sorted - d_min) / (d_max - d_min + EPSILON_STABILITY)

                x_sorted_tensor = torch.tensor(distorted_normalized, dtype=torch.float64).T.unsqueeze(0).to(device)
                x_raw_tensor = torch.tensor(data, dtype=torch.float64).unsqueeze(0).transpose(1, 2).to(device)
                
                with torch.no_grad():
                    if mode == 'ns':
                        predicted_alphas = model(x_sorted_tensor)
                        pred_a = predicted_alphas.cpu().numpy()[0]
                        pred_c = None 
                    else:
                        predicted_alphas, predicted_coefs = model(x_sorted_tensor, x_raw_tensor)
                        pred_a = predicted_alphas.cpu().numpy()[0]
                        pred_c = predicted_coefs.cpu().numpy()[0]
                        
                restored_data = np.zeros_like(data)
                for i in range(n_cols):
                    restored_data[:, i] = np.sign(data[:, i]) * (np.abs(data[:, i]) + BOX_COX_NEURAL_OFFSET) ** (1.0 / pred_a[i])

                self.frame.after(0, self._update_neural_ui, cols, n_cols, target_dim, pred_c, pred_a, restored_data, mode)
                
            except Exception as e:
                self.frame.after(0, lambda err=e: messagebox.showerror("Помилка PyTorch", str(err)))

        threading.Thread(target=_pytorch_worker, daemon=True).start()

    def _update_neural_ui(self, cols: List[str], n_cols: int, target_dim: int, predicted_coefs: Optional[np.ndarray], predicted_alphas: np.ndarray, restored_data: np.ndarray, mode: str):
        """Безпечна нитка оновлення інтерфейсу та виведення метрик нейромереж[cite: 20]."""
        self.neural_box.delete(1.0, tk.END)
        title = "Гібридний метод" if mode == 'ns' else "Нейромережевий метод"
        self.neural_box.insert(tk.END, f"=== ПЕРЕДБАЧЕННЯ ({title}) ===\n\n")
        self.neural_box.insert(tk.END, "=== ПАРАМЕТРИ ПЕРЕТВОРЕНЬ ===\n")
        if predicted_alphas is not None:
            self.neural_box.insert(tk.END, f"Коефіцієнти викривлення (CNN α): {np.round(predicted_alphas, 4)}\n")
        self.neural_box.insert(tk.END, "\n")
        self.neural_data_full = restored_data 
        
        corr_before = pd.DataFrame(restored_data, columns=cols).corr(method='pearson')
        self.neural_box.insert(tk.END, "=== КОРЕЛЯЦІЙНИЙ АНАЛІЗ ===\n")
        self.neural_box.insert(tk.END, "Матриця лінійної кореляції Пірсона (до фільтрації):\n")
        self._print_correlation_matrix(self.neural_box, corr_before, cols)

        self.neural_box.insert(tk.END, f"\n=== ФІЛЬТРАЦІЯ ({int(SIGMA_THRESHOLD)}-СІГМА) ===\n")
        z_scores = np.abs(stats.zscore(restored_data))
        mask = (z_scores < SIGMA_THRESHOLD).all(axis=1)
        self.neural_data_filtered = restored_data[mask]
        self.neural_box.insert(tk.END, f"Залишено точок: {len(self.neural_data_filtered)} з {len(restored_data)}\n\n")
        
        corr_after = pd.DataFrame(self.neural_data_filtered, columns=cols).corr(method='pearson')
        self.neural_box.insert(tk.END, "Матриця лінійної кореляції Пірсона (ПІСЛЯ фільтрації):\n")
        self._print_correlation_matrix(self.neural_box, corr_after, cols)
            
        if mode == 'ns':
            self.neural_coefs = self.apply_mnk(self.neural_data_filtered, cols, self.neural_box)
            self.neural_mean_vec = np.mean(self.neural_data_filtered, axis=0)
        else:
            self.neural_box.insert(tk.END, "Знайдено параметри:\n")
            for i in range(target_dim):
                self.neural_box.insert(tk.END, f"  a{i+1}: {predicted_coefs[i]:.6f}\n")
            self.neural_coefs = predicted_coefs
            self.neural_mean_vec = np.mean(self.neural_data_filtered, axis=0)

    def _build_plane_closure(self, mean_vec_attr: str, coefs_attr: str) -> Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]:
        """Фабрика функцій створення замикань генерації значень площин на графіку[cite: 20]."""
        def plane_function(x: np.ndarray, y: np.ndarray, idx_x: int, idx_y: int) -> np.ndarray:
            mean_v = getattr(self, mean_vec_attr)
            coefs = getattr(self, coefs_attr)
            z = mean_v[-1]
            if idx_x < len(coefs): 
                z += coefs[idx_x] * (x - mean_v[idx_x])
            if idx_y < len(coefs): 
                z += coefs[idx_y] * (y - mean_v[idx_y])
            return z
        return plane_function

    def vis_plane_comparison(self):
        """Агрегує поточні набори даних та надсилає подію побудови 3D-графіка площин[cite: 20]."""
        if not hasattr(self, 'bc_data_full') and not hasattr(self, 'neural_data_full'):
            return messagebox.showwarning("Увага", "Спочатку виконайте розрахунок хоча б одного методу.")
            
        state_str = self.plot_state_cb.get()
        cols, orig_data = self.get_selected_data()
        
        datasets_dict: Dict[str, np.ndarray] = {}
        planes: Dict[str, Tuple[Callable, str]] = {}
        
        if "1" in state_str:
            datasets_dict["До трансформації"] = orig_data
        elif "2" in state_str:
            if hasattr(self, 'bc_data_full') and self.bc_data_full is not None: 
                datasets_dict["Після Box-Cox"] = self.bc_data_full
            if hasattr(self, 'neural_data_full') and self.neural_data_full is not None: 
                datasets_dict["Після Neural CNN"] = self.neural_data_full
        else:
            if hasattr(self, 'bc_data_filtered') and self.bc_data_filtered is not None: 
                datasets_dict["Box-Cox (3-сігма)"] = self.bc_data_filtered
            if hasattr(self, 'neural_data_filtered') and self.neural_data_filtered is not None: 
                datasets_dict["Neural CNN (3-сігма)"] = self.neural_data_filtered
                
        if "1" not in state_str:
            if hasattr(self, 'analyt_coefs') and self.analyt_coefs is not None:
                planes["Аналітична площина"] = (self._build_plane_closure('analyt_mean_vec', 'analyt_coefs'), "blue")
                
            if hasattr(self, 'neural_coefs') and self.neural_coefs is not None:
                title = "Гібридний" if self.active_neural_mode == 'ns' else "Нейромережевий"
                planes[f"Площина ({title})"] = (self._build_plane_closure('neural_mean_vec', 'neural_coefs'), "red")

        self.app.show_plot(stage_num=5, datasets_dict=datasets_dict, title="Порівняння площин", show_plane=False, custom_planes=planes)

    def _calculate_residuals_and_r2(self, X: np.ndarray, Z_true: np.ndarray, mean_vec: np.ndarray, coefs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Допоміжний математичний метод розрахунку залишків регресії та R2[cite: 20]."""
        Z_pred = mean_vec[-1] + np.dot((X - mean_vec[:-1]), coefs)
        residuals = Z_true - Z_pred
        
        ss_tot = np.sum((Z_true - np.mean(Z_true)) ** 2)
        r2 = max(0.0, 1 - (np.sum(residuals ** 2) / ss_tot)) if ss_tot > 0 else 0.0
        return residuals, r2

    def _execute_clipboard_copy(self, text_widget: tk.Text):
        """Явний та безпечний інтерфейсний метод запису логів у буфер обміну ОС[cite: 20]."""
        self.frame.clipboard_clear()
        self.frame.clipboard_append(text_widget.get(1.0, tk.END))
        self.frame.update()

    def compare_methods(self):
        """Проводить перехресний аналіз точності моделей за допомогою критерію Стьюдента[cite: 20]."""
        if not hasattr(self, 'bc_data_full') or not hasattr(self, 'neural_data_full'):
            return messagebox.showwarning("Увага", "Спочатку виконайте розрахунок для ОБОХ методів.")
            
        mask_bc = (np.abs(stats.zscore(self.bc_data_full)) < SIGMA_THRESHOLD).all(axis=1)
        mask_nn = (np.abs(stats.zscore(self.neural_data_full)) < SIGMA_THRESHOLD).all(axis=1)
        common_mask = mask_bc & mask_nn
        
        bc_comp = self.bc_data_full[common_mask]
        nn_comp = self.neural_data_full[common_mask]
        
        # Обчислення метрик аналітичного контуру регресії
        res_analyt, r2_a = self._calculate_residuals_and_r2(
            bc_comp[:, :-1], bc_comp[:, -1], self.analyt_mean_vec, self.analyt_coefs
        )
        # Обчислення метрик нейромережевого контуру регресії
        res_neural, r2_n = self._calculate_residuals_and_r2(
            nn_comp[:, :-1], nn_comp[:, -1], self.neural_mean_vec, self.neural_coefs
        )
        
        abs_res_analyt = np.abs(res_analyt)
        abs_res_neural = np.abs(res_neural)
        
        mae_a, mae_n = np.mean(abs_res_analyt), np.mean(abs_res_neural)
        max_err_a, max_err_n = np.max(abs_res_analyt), np.max(abs_res_neural)
        
        t_stat_paired, p_val_paired = stats.ttest_rel(abs_res_analyt, abs_res_neural)
        
        # Ініціалізація UI форми звіту
        rep_win = tk.Toplevel(self.frame)
        neural_title = "Гібридний метод" if self.active_neural_mode == 'ns' else "Нейромережевий метод"
        rep_win.title(f"Порівняння: Аналітичний vs {neural_title}")
        rep_win.geometry("800x550")
        
        rep_text = tk.Text(rep_win, wrap=tk.WORD)
        rep_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        rep_text.insert(tk.END, f"{'Метрика ефективності':<28} | {'Аналітичний метод':<22} | {neural_title:<22}\n")
        rep_text.insert(tk.END, "-" * 78 + "\n")
        rep_text.insert(tk.END, f"{'MAE (Середня абсолютна)':<28} | {mae_a:<22.4f} | {mae_n:<22.4f}\n")
        rep_text.insert(tk.END, f"{'Max Error (Найгірша похибка)':<28} | {max_err_a:<22.4f} | {max_err_n:<22.4f}\n")
        rep_text.insert(tk.END, f"{'R² (Коеф. детермінації)':<28} | {r2_a:<22.4f} | {r2_n:<22.4f}\n\n")
        rep_text.insert(tk.END, f"Результати парного T-тесту похибок:\n  t-statistic = {t_stat_paired:.4f}\n  p-value = {p_val_paired:.4e}\n")

        ttk.Button(
            rep_win, 
            text=" Скопіювати звіт", 
            command=lambda: self._execute_clipboard_copy(rep_text)
        ).pack(fill=tk.X, padx=10, pady=5)

    def copy_all_results(self):
        """Збирає текстовий контент з усіх консолей виведення в єдиний буфер[cite: 20]."""
        stat_txt = self.stat_box.get(1.0, tk.END).strip()
        analyt_txt = self.analytical_box.get(1.0, tk.END).strip()
        neural_txt = self.neural_box.get(1.0, tk.END).strip()
        
        self.frame.clipboard_clear()
        self.frame.clipboard_append("\n\n".join([t for t in [stat_txt, analyt_txt, neural_txt] if t]))
        self.frame.update()