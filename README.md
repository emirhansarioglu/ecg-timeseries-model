# ecg-timeseries-model
A classifier for univariate echocardiogram (ECG) time series data. 

- **Dataset Exploration**: Conducted in `dataset_exploration/EDA.ipynb` with detailed plots and analysis to understand ECG signal patterns and distribution.
- **Modeling and Tuning**: Located in the `Modeling_Tuning_Augmentation/` folder, this includes:
  - Testing various RNN-based architectures (RNN, LSTM, GRU).
  - Hyperparameter tuning such as learning rate, dropout, and hidden layer size.
  - Evaluation using F1 Score for performance comparison.
- **Data Augmentation**: Techniques like noise injection, scaling, and time-shifting are applied and analyzed for performance impact.
- **Compression Techniques**: PCA and other dimensionality reduction methods are tested in `data_reduction/`, assessing reconstruction loss (MAE, MSE) at different compression levels.