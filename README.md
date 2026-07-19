# _Diploma_Bklvr: Interface Operations and Execution Guide

This document serves as a step-by-step operational manual for the graphical user interface of the surface modeling application. It explains the exact computational behavior and the underlying purpose of every button to allow any external user to replicate the entire data analysis and regression workflow.

## Data Control Actions

### Відкрити файл (TXT/CSV)
Clicking this button opens a system file dialog to import the raw text or comma-separated dataset. Mechanically, the application reads the file into a data frame, automatically finds text-based numeric fields, replaces any European decimal commas with standard dots, resets the calculation flags, and renders the raw dataset into the interactive table view. Use this action first to establish the working data environment.

### Видалити рядки з '0'/NaN
This button replaces all absolute zero values, zero strings, and missing data points with null indicators, deletes those rows completely from the dataset, and updates the view. It then displays a pop-up window showing the linear Pearson and Kendall rank correlation matrices computed before and after the erasure. Use this to prevent division-by-zero errors during subsequent mathematical power transformations and to evaluate how the removal of empty points impacts feature correlations.

### Зберегти зміни у файл
This button opens a save dialog allowing the user to export the current state of the modified, filtered, or transformed data table into a new CSV file. Use this to preserve clean or standardized datasets for external use or future modeling sessions.

### Скинути дані
Clicking this button completely discards all active transformations, shifts, standardizations, and regression outputs. It reloads the pure, unmodified dataset originally imported during the file load step. Use this if an experimental transformation sequence yields poor results and you need to restart the pipeline from scratch.

## Mathematical Transformation Buttons

### Зсув до додатніх
This button scans every column in the dataset to identify negative values or zeroes. If found, it automatically calculates the absolute minimum value of that specific column and adds that amount plus an additional offset of 0.01 to every element in the vector. This action is mandatory when using raw data with negative metrics because subsequent non-linear operations, like the Box-Cox transformation or neural power curves, require strictly positive inputs.

### Нормування (MaxAbs)
This button determines the maximum absolute value within each individual column and divides every entry in that column by that maximum value. This normalizes the data, strictly bounds all elements between -1.0 and 1.0, and sets the maximum absolute value to exactly 1.0. Use this to prevent columns with massive numerical scales from dominating the neural network features during gradient updates.

### Стандартизація (Z-score)
This button calculates the arithmetic mean and the standard deviation for every data column, subtracts the mean from each data point, and divides the result by the standard deviation. It then displays a summary window containing the exact calculated means and deviations for each feature. Use this to shift the data distribution to a zero mean and unit variance, which removes scale disparities across completely different physical measurements.

## Target Configuration and Variables Selection

### Залежна змінна (Z)
This drop-down combobox allows the user to declare which column represents the dependent target variable, which is the vertical axis or the output surface value to be predicted by the regression plane. By default, the application selects the final column of the loaded dataset.

### Незалежні змінні (X)
This multiple-selection listbox displays all available columns. The user must highlight one or more columns to act as the independent predictor variables or coordinate axes. The application automatically cross-checks the selections and ensures that whatever column is chosen as the dependent variable Z is excluded from the independent predictors X to avoid self-correlation.

## Processing and Core Method Modeling

### 1. Розрахувати статистику
Clicking this button evaluates the distribution shapes and statistical dependencies of the highlighted variables. It computes the minimum value, maximum value, arithmetic mean, standard deviation, skewness for asymmetry detection, and excess kurtosis for peak shape detection for each feature. If multiple features are highlighted, it also prints a structured Kendall rank correlation matrix to reveal non-linear monotonic relationships. The entire text report is rendered inside the left console window.

### 2. Аналітичний розв'язок
This button executes a three-stage classical mathematical modeling pipeline. First, it automatically calculates the ideal lambda exponents for the Box-Cox transformation to force the data into a normal distribution. Second, it computes Z-scores for the transformed entries and discards any rows where a value deviates by more than three standard deviations, executing the 3-sigma outlier filtration rule. Third, it calculates a centered Ordinary Least Squares linear regression to locate the precise coefficients of the analytical plane. The results, including transformation parameters, pre-filter and post-filter Pearson correlation matrices, remaining sample counts, and final plane coefficients, are pushed into the central console window.

### 3. Гібридний метод (.pt)
This button prompts the user to select a pre-trained PyTorch weight file for the CurvatureCNN architecture. The application launches a separate background computing thread to prevent the interface from freezing. The thread sorts and normalizes the data, feeds it through the convolutional layers to estimate optimal space curvature coefficients, performs an inverse power-based space restoration, applies the 3-sigma filtration rule, and computes the plane parameters using the ordinary least squares method. The resulting CNN curvature values, Pearson matrices, and calculated coefficients are sent directly to the right console window. Use this to combine deep learning geometric corrections with classical linear optimization.

### 3. Нейромережевий метод (.pt)
This button functions similarly to the hybrid approach but loads the PureNeuralModel architecture. The background computing thread utilizes the CurvatureCNN block to predict space distortion metrics and then routes the restored data directly through the CoefRegressor multi-layer perceptron to predict the final plane coefficients natively through the network graph without relying on secondary least-squares operations. The output parameters and data matrices are displayed in the right console window. Use this for a pure, end-to-end neural network regression inference.

## Verification and Clipboard Tools

### 4. Візуальне порівняння площин
This button references the configuration dropdown directly to its left, which specifies whether to display original data, transformed data, or 3-sigma filtered data. It then calls the primary application controller to launch an interactive 3D visualization window. This window maps the selected data points as a spatial cloud and overlays the analytical regression surface in blue and the neural regression surface in red. Use this tool to visually evaluate how closely each model matches the geometric distribution of the data points.

### 5. Порівняння методів (T-test Стьюдента)
This button isolates the subset of data points that successfully survived the outlier filtering filters of both modeling pipelines. It evaluates the exact true target value against the predicted values for both the analytical and neural models to calculate the Mean Absolute Error, the worst-case Maximum Error, and the R-squared coefficient of determination for both methods. Finally, it executes a paired Student's t-test on the absolute residuals of both models to output a p-value indicating whether the accuracy difference between the two systems is statistically significant. The report appears in a new standalone top-level interface window.

### Скопіювати усі результати
Clicking this button reads the plain text contents from the statistics box, the analytical box, and the neural box simultaneously. It filters out empty sections, joins the available text blocks with double line breaks, and pushes the entire consolidated report directly into the operating system clipboard. Use this action to immediately paste all execution metrics, coefficients, and statistical test values into your thesis document or spreadsheet software.
