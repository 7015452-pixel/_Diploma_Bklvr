import tkinter as tk
from tkinter import ttk
import numpy as np

# Імпорти внутрішніх модулів проєкту
from stage_generation import CombinedGenUI
from stage_x_analitics import Stage3UI
from stage_y_neuron import Stage4UI
from plotter import PlotWindow
from stage_z_file import FileTabUI
from analysis_tab import AnalysisTabUI

# Налаштування головного вікна та інтерфейсу
WINDOW_TITLE = "Моделювання: Від МГК до Нейромереж (2026)"
WINDOW_GEOMETRY = "1100x850"
FRAME_PADDING = 10
PANEL_PADDING = 5


class RegressionApp:
    """Головний клас програми для регресійного моделювання та аналізу."""

    def __init__(self, root: tk.Tk):
        """Ініціалізує головне вікно та спільні дані додатка."""
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)

        # Спільний стан даних між етапами
        self.stage1_data = None
        self.stage2_data = None
        self.stage3_mgk_data = None
        self.stage3_bc_data = None
        self.true_coefs = None
        
        self.stage1_shifts = np.zeros(3)
        self.feature_names = ["X1", "X2", "X3"]
        self.plot_windows = {}

        # Історія навчання нейромережі
        self.training_history = {
            "epoch": [],
            "loss": [],
            "alpha_error": []
        }

        self.setup_ui()

    def setup_ui(self):
        """Створює та налаштовує графічний інтерфейс програми."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=PANEL_PADDING, pady=PANEL_PADDING)

        self.tab_generation = ttk.Frame(self.notebook)
        self.tab_file = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_generation, text="Генерація")
        self.notebook.add(self.tab_file, text="Файл")
        self.notebook.add(self.tab_analysis, text="Аналіз і графіки")

        # Панель вкладки генерації та моделювання
        main_frame = ttk.Frame(self.tab_generation, padding=FRAME_PADDING)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=PANEL_PADDING)

        self.gen_frame = ttk.LabelFrame(left_panel, text="Етап 1 & 2: Підготовка даних", padding=FRAME_PADDING)
        self.gen_frame.pack(fill=tk.BOTH, expand=True)
        self.generation = CombinedGenUI(self.gen_frame, self)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=PANEL_PADDING)

        self.stage3_frame = ttk.LabelFrame(right_panel, text="Етап 3: Алгебраїчний підхід (МГК + Box-Cox)", padding=FRAME_PADDING)
        self.stage3_frame.pack(fill=tk.BOTH, expand=True, pady=(0, PANEL_PADDING))
        self.stage3 = Stage3UI(self.stage3_frame, self)

        self.stage4_frame = ttk.LabelFrame(right_panel, text="Етап 4: Інтелектуальний підхід (MLP)", padding=FRAME_PADDING)
        self.stage4_frame.pack(fill=tk.BOTH, expand=True)
        self.stage4 = Stage4UI(self.stage4_frame, self)

        # Ініціалізація допоміжних вкладок
        self.file_ui = FileTabUI(self.tab_file, self)
        self.analysis_ui = AnalysisTabUI(self.tab_analysis, self)

    def show_plot(self, stage_num: int, datasets_dict: dict, title: str, show_plane: bool = False, custom_planes: dict = None):
        """Відображає або оновлює вікно з графіками для обраного етапу."""
        window_exists = stage_num in self.plot_windows and tk.Toplevel.winfo_exists(self.plot_windows[stage_num])

        if not window_exists:
            self.plot_windows[stage_num] = PlotWindow(self.root, title)
        else:
            self.plot_windows[stage_num].title(title)

        # Для другого та третього етапів площина відображається завжди
        show_plane = show_plane or stage_num in (2, 3)
        planes = {}

        if show_plane and self.true_coefs is not None:
            planes["Ідеал (Stage 1)"] = (self.get_ideal_plane_fn(), "red")

        if custom_planes:
            planes.update(custom_planes)

        self.plot_windows[stage_num].set_datasets(datasets_dict, self.feature_names, planes)

    def get_ideal_plane_fn(self):
        """Повертає функцію обчислення координат ідеальної площини першого етапу."""
        def fn(x, y, idx_x, idx_y):
            shifts = self.stage1_shifts
            coefs = self.true_coefs
            z = shifts[-1]

            if idx_x < len(coefs):
                z += coefs[idx_x] * (x - shifts[idx_x])

            if idx_y < len(coefs):
                z += coefs[idx_y] * (y - shifts[idx_y])

            return z

        return fn


if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()