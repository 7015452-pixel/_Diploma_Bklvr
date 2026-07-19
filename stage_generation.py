import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.stats as stats

# Параметри геометричного розміщення та інтерфейсу
FRAME_PADDING = 5
GRID_PADX = 5
GRID_PADY = 2
BUTTON_PADDING = 2
ALPHA_ENTRY_WIDTH = 5
TEXT_HEIGHT = 15


class CombinedGenUI:
    """Вкладка для генерації лінійної бази даних та її степеневого викривлення."""

    def __init__(self, frame: tk.Widget, app):
        """Створює елементи керування для етапів генерації та трансформації."""
        self.app = app
        self.frame = frame

        # Етап 1: Конфігурація лінійної бази
        gen_box = ttk.LabelFrame(
            self.frame, 
            text="Етап 1: Генерація лінійної бази", 
            padding=FRAME_PADDING
        )
        gen_box.pack(fill=tk.X, pady=FRAME_PADDING)
        gen_box.columnconfigure(1, weight=1)

        ttk.Label(gen_box, text="Коефіцієнти a1, a2:").grid(row=0, column=0, sticky="w")
        self.entry_coefs = ttk.Entry(gen_box)
        self.entry_coefs.insert(0, "2.0, -1.5")
        self.entry_coefs.grid(row=0, column=1, sticky="ew", padx=GRID_PADX, pady=GRID_PADY)

        ttk.Label(gen_box, text="Кількість точок (N):").grid(row=1, column=0, sticky="w")
        self.entry_n = ttk.Entry(gen_box)
        self.entry_n.insert(0, "500")
        self.entry_n.grid(row=1, column=1, sticky="ew", padx=GRID_PADX, pady=GRID_PADY)

        ttk.Label(gen_box, text="Шум (Sigma):").grid(row=2, column=0, sticky="w")
        self.entry_sigma = ttk.Entry(gen_box)
        self.entry_sigma.insert(0, "0.01")
        self.entry_sigma.grid(row=2, column=1, sticky="ew", padx=GRID_PADX, pady=GRID_PADY)

        self.cb_noise_mode = ttk.Combobox(gen_box, values=["Тільки X3", "Всі осі"], state="readonly")
        self.cb_noise_mode.current(0)
        self.cb_noise_mode.grid(row=3, column=0, columnspan=2, sticky="ew", pady=GRID_PADY)

        opt_frame = ttk.Frame(gen_box)
        opt_frame.grid(row=4, column=0, columnspan=2, pady=FRAME_PADDING)
        
        self.norm_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Нормування площини", variable=self.norm_enabled).pack(side=tk.LEFT, padx=10)
        
        self.shift_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Зсув (+0.01)", variable=self.shift_enabled).pack(side=tk.LEFT, padx=10)

        self.epsilon_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Шум", variable=self.epsilon_enabled).pack(side=tk.LEFT, padx=10)

        btn_frame1 = ttk.Frame(gen_box)
        btn_frame1.grid(row=5, column=0, columnspan=2, sticky="ew", pady=FRAME_PADDING)
        
        ttk.Button(
            btn_frame1, 
            text="Згенерувати базу", 
            command=self.generate_base
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, BUTTON_PADDING))
        
        ttk.Button(
            btn_frame1, 
            text="Візуалізувати (Етап 1)", 
            command=self.visualize_stage1
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(BUTTON_PADDING, 0))

        # Етап 2: Степеневе викривлення
        trans_box = ttk.LabelFrame(
            self.frame, 
            text="Етап 2: Степеневе викривлення (X^α)", 
            padding=FRAME_PADDING
        )
        trans_box.pack(fill=tk.X, pady=FRAME_PADDING)

        rnd_frame = ttk.Frame(trans_box)
        rnd_frame.pack(fill=tk.X, pady=GRID_PADY)
        
        ttk.Label(rnd_frame, text="Min α:").pack(side=tk.LEFT)
        self.entry_alpha_min = ttk.Entry(rnd_frame, width=ALPHA_ENTRY_WIDTH)
        self.entry_alpha_min.insert(0, "1.5")
        self.entry_alpha_min.pack(side=tk.LEFT, padx=GRID_PADX)
        
        ttk.Label(rnd_frame, text="Max α:").pack(side=tk.LEFT)
        self.entry_alpha_max = ttk.Entry(rnd_frame, width=ALPHA_ENTRY_WIDTH)
        self.entry_alpha_max.insert(0, "3.0")
        self.entry_alpha_max.pack(side=tk.LEFT, padx=GRID_PADX)

        ttk.Label(trans_box, text="Власні α (1 або 3 через кому):").pack(anchor="w", pady=(FRAME_PADDING, 0))
        self.entry_manual_alpha = ttk.Entry(trans_box)
        self.entry_manual_alpha.insert(0, "2.0, 0.5, 1.5")
        self.entry_manual_alpha.pack(fill=tk.X, pady=GRID_PADY)

        self.cb_alpha_mode = ttk.Combobox(
            trans_box, 
            values=["Випадково: Одне α для всіх", "Випадково: Три різні α", "Вручну"], 
            state="readonly"
        )
        self.cb_alpha_mode.current(0)
        self.cb_alpha_mode.pack(fill=tk.X, pady=FRAME_PADDING)

        btn_frame2 = ttk.Frame(trans_box)
        btn_frame2.pack(fill=tk.X)
        
        ttk.Button(
            btn_frame2, 
            text="Трансформувати", 
            command=self.apply_power
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, BUTTON_PADDING))
        
        ttk.Button(
            btn_frame2, 
            text="Скинути", 
            command=self.reset_data
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(BUTTON_PADDING, BUTTON_PADDING))
        
        ttk.Button(
            btn_frame2, 
            text="Візуалізувати", 
            command=self.visualize_stage2
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(BUTTON_PADDING, 0))

        self.info_text = tk.Text(self.frame, height=TEXT_HEIGHT)
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=FRAME_PADDING)

    def log_statistics(self, data: np.ndarray, title: str):
        """Обчислює та виводить основні статистичні показники масиву даних."""
        n_cols = data.shape[1]
        msg = f"\n--- Статистика ({title}) ---\n"
        feature_names = getattr(self.app, "feature_names", None)

        for i in range(n_cols):
            col_data = data[:, i]
            min_v = np.min(col_data)
            max_v = np.max(col_data)
            mean_v = np.mean(col_data)
            std_v = np.std(col_data)
            skew_v = stats.skew(col_data)
            kurt_v = stats.kurtosis(col_data)
            
            if feature_names and len(feature_names) > i:
                axis_name = feature_names[i]
            else:
                axis_name = f"X{i+1}"
                
            msg += f"[{axis_name}]: Min={min_v:.3f}, Max={max_v:.3f}, Mean={mean_v:.3f}, Std={std_v:.3f}, Skew={skew_v:.3f}, Kurt={kurt_v:.3f}\n"
        
        self.update_info(msg.strip())

    def generate_base(self):
        """Генерує набір даних лінійної бази із заданими параметрами та шумом."""
        self.info_text.delete(1.0, tk.END)
        try:
            coefs = np.array([float(x.strip()) for x in self.entry_coefs.get().split(',')])
            n = int(self.entry_n.get())
            sigma = float(self.entry_sigma.get())
            n_features = len(coefs)
            
            X_indep = np.random.normal(0, 1.0, (n, n_features))
            X_dep = np.dot(X_indep, coefs)
            data = np.column_stack((X_indep, X_dep))
            self.app.true_coefs = coefs

            if self.norm_enabled.get():
                lmb_inv = np.sqrt(1 + np.sum(coefs**2))
                data = data * (1.0 / lmb_inv)

            num_dimensions = data.shape[1]
            self.app.stage1_shifts = np.zeros(num_dimensions)
            
            if self.shift_enabled.get():
                for i in range(num_dimensions):
                    y_max = np.max(np.abs(data[:, i]))
                    data[:, i] += y_max
                    self.app.stage1_shifts[i] = y_max

            if self.epsilon_enabled.get():
                if self.cb_noise_mode.get() == "Всі осі":
                    noise = np.random.normal(0, sigma, data.shape)
                    data += noise
                else:
                    noise = np.random.normal(0, sigma, n)
                    data[:, -1] += noise

            feature_names = [f"X{i+1}" for i in range(num_dimensions - 1)] + [f"Z(X{num_dimensions})"]
            self.app.feature_names = feature_names
            self.app.stage1_data = data
            self.app.stage2_data = np.copy(data)
            
            status_msg = (
                f"=== Етап 1 Завершено ===\n"
                f"N={n}, Вимірів={num_dimensions}, Coefs={coefs}\n"
                f"Нормування: {self.norm_enabled.get()} | Зсув: {self.shift_enabled.get()} | Шум: {self.epsilon_enabled.get()}"
            )
            self.update_info(status_msg)
            self.log_statistics(data, "Лінійна база")
            
        except Exception as e: 
            messagebox.showerror("Помилка", str(e))

    def apply_power(self):
        """Застосовує обраний режим степеневого викривлення до масиву даних."""
        source_data = getattr(self.app, "stage2_data", None)
        if source_data is None:
            source_data = getattr(self.app, "stage1_data", None)
            
        if source_data is None:
            messagebox.showwarning("Увага", "Спочатку згенеруйте базу (Етап 1)!")
            return

        try:
            n_cols = source_data.shape[1]
            idx = self.cb_alpha_mode.current()

            if idx in [0, 1]:  
                min_a = float(self.entry_alpha_min.get())
                max_a = float(self.entry_alpha_max.get())

                if min_a >= max_a:
                    raise ValueError("Мінімальне значення має бути меншим за максимальне.")

                if idx == 0:  
                    single_alpha = np.random.uniform(min_a, max_a)
                    alphas = [single_alpha] * n_cols
                else:         
                    alphas = np.random.uniform(min_a, max_a, size=n_cols)
            else:  
                manual_str = self.entry_manual_alpha.get()
                parsed_alphas = [float(x.strip()) for x in manual_str.split(',')]
                
                if len(parsed_alphas) == 1:
                    alphas = [parsed_alphas[0]] * n_cols
                elif len(parsed_alphas) == n_cols:
                    alphas = parsed_alphas
                else:
                    raise ValueError(f"Потрібно ввести 1 або {n_cols} значень через кому.")
            
            transformed_data = np.zeros_like(source_data)
            for i in range(n_cols):
                transformed_data[:, i] = np.sign(source_data[:, i]) * (np.abs(source_data[:, i]) ** alphas[i])
            
            self.app.stage2_alphas = alphas  
            self.app.true_alphas = alphas
            self.app.stage2_data = transformed_data
            
            msg = "=== Етап 2: Трансформація ===\nЗастосовано α до осей:\n"
            feature_names = getattr(self.app, "feature_names", None)
            
            for i, a in enumerate(alphas):
                if feature_names and len(feature_names) > i:
                    axis_name = feature_names[i]
                else:
                    axis_name = f"X{i+1}"
                msg += f"Вісь {axis_name}: α = {a:.4f}\n"
                
            self.update_info(msg.strip())
            self.log_statistics(transformed_data, "Після X^α")
            
        except Exception as e: 
            messagebox.showerror("Помилка", f"Перевеірте введені дані. {e}")

    def reset_data(self):
        """Скасовує степеневі зміни та повертає дані до вихідного лінійного стану."""
        stage1_data = getattr(self.app, "stage1_data", None)
        
        if stage1_data is not None:
            self.app.stage2_data = np.copy(stage1_data)
            self.update_info("=== Скидання ===\nДані повернуто до початкового лінійного стану (Етап 1).")
        else:
            messagebox.showwarning("Увага", "Немає початкових даних з Етапу 1!")
            
    def visualize_stage1(self):
        """Ініціює побудову графіка для лінійної бази даних."""
        stage1_data = getattr(self.app, "stage1_data", None)
        
        if stage1_data is None: 
            messagebox.showwarning("Увага", "Немає даних. Згенеруйте Етап 1.")
            return
            
        self.app.show_plot(1, {"Дані (Етап 1)": stage1_data}, "Етап 1", show_plane=True)

    def visualize_stage2(self):
        """Ініціює побудову графіка для викривлених даних."""
        stage2_data = getattr(self.app, "stage2_data", None)
        
        if stage2_data is None: 
            messagebox.showwarning("Увага", "Немає даних. Застосуйте трансформацію.")
            return
            
        self.app.show_plot(2, {"Трансформовані дані (X^α)": stage2_data}, "Етап 2: Степеневе перетворення", show_plane=True)

    def update_info(self, msg: str):
        """Додає текстове повідомлення в інформаційне вікно інтерфейсу."""
        self.info_text.insert(tk.END, msg + "\n" + "-"*40 + "\n")
        self.info_text.see(tk.END)