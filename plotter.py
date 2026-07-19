import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Callable

# Налаштування вікна та параметри візуалізації
WINDOW_GEOMETRY = "850x800"
FIG_SIZE = (7, 6)
FIGURE_DPI = 100

# Параметри 3D відображення
DEFAULT_ELEV = 30
DEFAULT_AZIM = -60
ROTATION_STEP = 5
GRID_RESOLUTION = 20

# Налаштування прозорості та маркерів
ALPHA_SCATTER_3D = 0.6
ALPHA_SURFACE_3D = 0.4
ALPHA_SCATTER_MATRIX = 0.5
ALPHA_HIST = 0.7


class PlotWindow(tk.Toplevel):
    """Вікно відображення графіків із підтримкою 3D хмари точок та матриці парних розподілів (Pairplot)."""

    def __init__(self, parent: tk.Widget, title: str):
        """Ініціалізує компоненти графічного інтерфейсу та обробники подій обертання площини."""
        super().__init__(parent)
        self.title(title)
        self.geometry(WINDOW_GEOMETRY) 
        
        # Стан даних вікна
        self.datasets: Dict[str, np.ndarray] = {}
        self.current_data_key: str = ""
        self.feature_names: List[str] = []
        self.planes: Dict[str, Tuple[Callable, str]] = {} 
        self.plane_vars: Dict[str, tk.BooleanVar] = {} 
        self.plane_checkboxes: List[ttk.Checkbutton] = []

        # Початкові кути огляду 3D
        self.elev = DEFAULT_ELEV
        self.azim = DEFAULT_AZIM
        self.ax = None

        # Режим відображення: "3d" або "matrix"
        self.view_mode = tk.StringVar(value="3d")

        # Головні контейнери інтерфейсу
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = plt.Figure(figsize=FIG_SIZE, dpi=FIGURE_DPI)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10, padx=10)

        # Панель вибору набору даних та режиму
        self.dataset_frame = ttk.Frame(self.control_frame)
        self.dataset_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(self.dataset_frame, text="Вибір даних:").pack(side=tk.LEFT)
        self.cb_dataset = ttk.Combobox(self.dataset_frame, state="readonly", width=40)
        self.cb_dataset.pack(side=tk.LEFT, padx=10)
        self.cb_dataset.bind("<<ComboboxSelected>>", self.on_dataset_change)

        ttk.Radiobutton(
            self.dataset_frame, text="3D Хмара", variable=self.view_mode, value="3d", command=self.draw_plot
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(
            self.dataset_frame, text="Матриця (Pairplot)", variable=self.view_mode, value="matrix", command=self.draw_plot
        ).pack(side=tk.LEFT)

        # Панелі керування осями та площинами
        self.axes_frame = ttk.Frame(self.control_frame)
        self.axes_frame.pack(side=tk.LEFT, padx=10)

        self.planes_frame = ttk.Frame(self.control_frame)
        self.planes_frame.pack(side=tk.LEFT, padx=20)

        self.cb_x = ttk.Combobox(self.axes_frame, state="readonly", width=10)
        self.cb_y = ttk.Combobox(self.axes_frame, state="readonly", width=10)
        self.cb_z = ttk.Combobox(self.axes_frame, state="readonly", width=10)

        ttk.Label(self.axes_frame, text="Вісь X:").grid(row=0, column=0, sticky="e")
        self.cb_x.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.axes_frame, text="Вісь Y:").grid(row=1, column=0, sticky="e")
        self.cb_y.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(self.axes_frame, text="Вісь Z:").grid(row=2, column=0, sticky="e")
        self.cb_z.grid(row=2, column=1, padx=5, pady=2)

        self.cb_x.bind("<<ComboboxSelected>>", lambda e: self.draw_plot())
        self.cb_y.bind("<<ComboboxSelected>>", lambda e: self.draw_plot())
        self.cb_z.bind("<<ComboboxSelected>>", lambda e: self.draw_plot())

        # Прив'язка клавіш для керування камерою 3D
        self.bind("<Up>", lambda e: self._rotate_camera(d_elev=ROTATION_STEP))
        self.bind("<Down>", lambda e: self._rotate_camera(d_elev=-ROTATION_STEP))
        self.bind("<Left>", lambda e: self._rotate_camera(d_azim=-ROTATION_STEP))
        self.bind("<Right>", lambda e: self._rotate_camera(d_azim=ROTATION_STEP))
        self.focus_set()

    def set_datasets(self, datasets: Dict[str, np.ndarray], names: List[str], planes: Dict[str, Tuple[Callable, str]] = None):
        """Оновлює внутрішні масиви даних, назви ознак та пересоздає елементи керування площинами."""
        self.datasets = datasets
        self.feature_names = names
        self.planes = planes if planes else {}
        
        # Очищення застарілих чекбоксів з цільового фрейму
        for cb in self.plane_checkboxes:
            cb.destroy()
        self.plane_checkboxes.clear()
        self.plane_vars.clear()

        # Генерація чекбоксів для відображення регресійних площин
        for name in self.planes.keys():
            var = tk.BooleanVar(value=True)
            self.plane_vars[name] = var
            cb = ttk.Checkbutton(self.planes_frame, text=name, variable=var, command=self.draw_plot)
            cb.pack(side=tk.LEFT, padx=10)
            self.plane_checkboxes.append(cb)

        # Заповнення комбобоксу вибору активного датасету
        self.cb_dataset['values'] = list(self.datasets.keys())
        if self.datasets:
            self.cb_dataset.current(0)
            self.current_data_key = self.cb_dataset.get()
                
        # Налаштування доступних вимірів для осей координат
        if self.feature_names:
            self.cb_x['values'] = self.feature_names
            self.cb_y['values'] = self.feature_names
            self.cb_z['values'] = self.feature_names
            
            if len(self.feature_names) >= 3:
                self.cb_x.current(0)
                self.cb_y.current(1)
                self.cb_z.current(2)
        
        self.draw_plot()

    def on_dataset_change(self, event: tk.Event):
        """Обробник події зміни вихідного набору даних."""
        self.current_data_key = self.cb_dataset.get()
        self.draw_plot()

    def draw_plot(self):
        """Перемальовує графічне вікно відповідно до встановленого режиму огляду."""
        if not self.current_data_key or self.current_data_key not in self.datasets: 
            return
            
        data = self.datasets[self.current_data_key]
        idx_x = self.cb_x.current()
        idx_y = self.cb_y.current()
        idx_z = self.cb_z.current()

        if idx_x < 0 or idx_y < 0 or idx_z < 0: 
            return

        self.figure.clear()

        if self.view_mode.get() == "3d":
            self.draw_3d(data, idx_x, idx_y, idx_z)
        else:
            self.draw_matrix(data, idx_x, idx_y, idx_z)
            
        self.canvas.draw()

    def draw_3d(self, data: np.ndarray, idx_x: int, idx_y: int, idx_z: int):
        """Будує тривимірний точковий графік та відповідні апроксимуючі площини."""
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        x_data = data[:, idx_x]
        y_data = data[:, idx_y]
        z_data = data[:, idx_z]

        self.ax.scatter(x_data, y_data, z_data, c='blue', marker='o', alpha=ALPHA_SCATTER_3D)
        last_idx = data.shape[1] - 1
        
        # Відображення площин при правильній комбінації осей (залежна координата на осі Z)
        is_valid_axes = (idx_z == last_idx) and (idx_x != last_idx) and (idx_y != last_idx)
        if self.planes and is_valid_axes:
            x_range = np.linspace(min(x_data), max(x_data), GRID_RESOLUTION)
            y_range = np.linspace(min(y_data), max(y_data), GRID_RESOLUTION)
            x_grid, y_grid = np.meshgrid(x_range, y_range)

            for name, var in self.plane_vars.items():
                if var.get():
                    fn, color = self.planes[name]
                    if fn:
                        z_grid = fn(x_grid, y_grid, idx_x, idx_y)
                        self.ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=ALPHA_SURFACE_3D)

        self.ax.set_xlabel(self.feature_names[idx_x])
        self.ax.set_ylabel(self.feature_names[idx_y])
        self.ax.set_zlabel(self.feature_names[idx_z])
        self.ax.set_title(f"{self.current_data_key} (3D)")
        self.ax.view_init(elev=self.elev, azim=self.azim)

    def draw_matrix(self, data: np.ndarray, idx_x: int, idx_y: int, idx_z: int):
        """Генерує кореляційну матрицю ознак, гістограми та статистичні метрики 분포."""
        data_3d = data[:, [idx_x, idx_y, idx_z]]
        names = [self.feature_names[idx_x], self.feature_names[idx_y], self.feature_names[idx_z]]
        
        # Обчислення кількості бінів за емпіричним правилом Стерджеса
        n_samples = len(data_3d)
        m_bins = int(np.ceil(1 + 3.322 * np.log10(n_samples)))
        
        axs = self.figure.subplots(3, 3)
        self.figure.suptitle(f"{self.current_data_key} (Аналіз пар | N={n_samples}, Стовпчиків M={m_bins})", fontsize=12)

        diag_pairs = [(0, 1), (1, 2), (2, 0)]
        upper_triangle_map = {(0, 1): 0, (0, 2): 1, (1, 2): 2}

        for i in range(3):
            for j in range(3):
                ax = axs[i, j]
                
                # Головна діагональ: Кореляційні поля взаємозв'язку
                if i == j:
                    px, py = diag_pairs[i]
                    ax.scatter(data_3d[:, px], data_3d[:, py], alpha=ALPHA_SCATTER_MATRIX, s=5, c='purple')
                    ax.set_xlabel(names[px], fontsize=8)
                    ax.set_ylabel(names[py], fontsize=8)
                    ax.set_title(f"{names[px]} vs {names[py]}", fontsize=9)
                
                # Верхній трикутник: Одновимірні розподіли (Гістограми відносних частот)
                elif i < j:
                    var_idx = upper_triangle_map[(i, j)]
                    current_data = data_3d[:, var_idx]
                    weights = np.ones_like(current_data) / len(current_data)
                    
                    ax.hist(current_data, bins=m_bins, weights=weights, color='skyblue', edgecolor='black', alpha=ALPHA_HIST)
                    ax.set_title(f"Розподіл {names[var_idx]}", fontsize=9)
                    ax.set_xlabel(names[var_idx], fontsize=8)
                    ax.set_ylabel("Відносна частота", fontsize=8)
                
                # Нижній трикутник: Текстовий опис статистичних характеристик
                else:
                    var_x = data_3d[:, j]
                    var_y = data_3d[:, i]
                    
                    corr, _ = stats.pearsonr(var_x, var_y)
                    skew_x = stats.skew(var_x)
                    kurt_x = stats.kurtosis(var_x)
                    skew_y = stats.skew(var_y)
                    kurt_y = stats.kurtosis(var_y)
                    
                    text = (
                        f"Corr (r) = {corr:.2f}\n\n"
                        f"{names[j]}:\n"
                        f"Skew: {skew_x:.2f} | Kurt: {kurt_x:.2f}\n\n"
                        f"{names[i]}:\n"
                        f"Skew: {skew_y:.2f} | Kurt: {kurt_y:.2f}"
                    )
                    
                    ax.text(
                        0.5, 0.5, text, ha='center', va='center', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

        self.figure.tight_layout()

    def _rotate_camera(self, d_elev: int = 0, d_azim: int = 0):
        """Внутрішній уніфікований метод зміни кута огляду 3D сцени."""
        if self.view_mode.get() == "3d" and self.ax is not None:
            self.elev += d_elev
            self.azim += d_azim
            self.ax.view_init(elev=self.elev, azim=self.azim)
            self.canvas.draw()