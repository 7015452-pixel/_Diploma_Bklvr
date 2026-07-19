import tkinter as tk
from tkinter import ttk, messagebox
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Параметри побудови графіків та інтерфейсу
DEFAULT_FIGSIZE = (8, 5)
DEFAULT_DPI = 100
FRAME_PADDING = 10
PLOT_PADDING = 5


class AnalysisTabUI:
    """Вкладка для аналізу та візуалізації графіків навчання."""

    def __init__(self, parent: tk.Widget, app):
        """Створює вкладку аналізу."""
        self.app = app
        self.frame = parent
        self.plot_canvas = None
        self.current_figure = None

        control_frame = ttk.LabelFrame(
            self.frame,
            text="Аналіз навчання нейромереж",
            padding=FRAME_PADDING
        )
        control_frame.pack(fill=tk.X, padx=PLOT_PADDING, pady=PLOT_PADDING)

        ttk.Button(
            control_frame,
            text="Графік Loss",
            command=self.show_loss
        ).pack(side=tk.LEFT, padx=PLOT_PADDING)

        ttk.Button(
            control_frame,
            text="Графік Alpha Error",
            command=self.show_alpha_error
        ).pack(side=tk.LEFT, padx=PLOT_PADDING)

        ttk.Button(
            control_frame,
            text="Графік a_i",
            command=self.show_ai
        ).pack(side=tk.LEFT, padx=PLOT_PADDING)

        ttk.Button(
            control_frame,
            text="Очистити графіки",
            command=self.clear_plot
        ).pack(side=tk.LEFT, padx=PLOT_PADDING)

        self.plot_frame = ttk.Frame(self.frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def clear_plot(self):
        """Очищає поточний графік."""
        if self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None

        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None

    def draw_plot(self, x_data: Sequence, y_data: Sequence, title: str, ylabel: str):
        """Будує графік за заданими даними."""
        self.clear_plot()

        figure = plt.Figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
        self.current_figure = figure

        axes = figure.add_subplot(111)
        axes.plot(x_data, y_data)

        axes.set_title(title)
        axes.set_xlabel("Ітерація (Епоха)")
        axes.set_ylabel(ylabel)
        axes.grid(True)

        self.plot_canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_loss(self):
        """Відображає графік функції втрат."""
        training_history = getattr(self.app, "training_history", None)

        if not training_history or len(training_history["epoch"]) == 0:
            messagebox.showwarning(
                "Увага",
                "Немає даних навчання. Запустіть навчання на 4-му етапі."
            )
            return

        self.draw_plot(
            training_history["epoch"],
            training_history["loss"],
            "Функція втрат (Loss)",
            "MSE"
        )

    def show_alpha_error(self):
        """Відображає графік похибки параметра alpha."""
        training_history = getattr(self.app, "training_history", None)

        if not training_history or len(training_history["epoch"]) == 0:
            messagebox.showwarning(
                "Увага",
                "Немає даних навчання. Запустіть навчання на 4-му етапі."
            )
            return

        self.draw_plot(
            training_history["epoch"],
            training_history["alpha_error"],
            "Похибка α",
            "Абсолютна похибка"
        )

    def show_ai(self):
        """Відображає графік похибки параметрів a_i."""
        training_history = getattr(self.app, "training_history", None)

        if not training_history or len(training_history["epoch"]) == 0:
            messagebox.showwarning(
                "Увага",
                "Немає даних навчання. Запустіть навчання на 4-му етапі."
            )
            return

        self.draw_plot(
            training_history["epoch"],
            training_history["coef_error"],
            "Похибка параметрів aᵢ",
            "MAE(a_i)"
        )