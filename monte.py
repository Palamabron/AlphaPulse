"""
Advanced Validation Framework for Numerai
- Walk-Forward Analysis
- Monte Carlo Simulation
- Normal Distribution Testing
- Statistical Risk Assessment
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import lightgbm as lgb
import cloudpickle

from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution


# ============================================================================
# 1. WALK-FORWARD ANALYSIS (Rolling Window Validation)
# ============================================================================

class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis для Numerai
    Симулирует реальный процесс: обучение → предсказание → сдвиг окна
    """

    def __init__(self, train_window_size: int = 200,
                 test_window_size: int = 20,
                 step_size: int = 20):
        """
        Args:
            train_window_size: размер окна обучения (в эрах)
            test_window_size: размер окна тестирования (в эрах)
            step_size: шаг сдвига окна (в эрах)
        """
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.step_size = step_size

    def create_walk_forward_splits(self, eras: np.ndarray) -> List[Tuple]:
        """
        Создает walk-forward splits

        Example:
        eras = [0001, 0002, ..., 0573]

        Split 1: Train [0001-0200], Test [0201-0220]
        Split 2: Train [0021-0220], Test [0221-0240]
        Split 3: Train [0041-0240], Test [0241-0260]
        ...
        """
        unique_eras = sorted(np.unique(eras))
        n_eras = len(unique_eras)

        splits = []
        start_idx = 0

        while start_idx + self.train_window_size + self.test_window_size <= n_eras:
            train_end = start_idx + self.train_window_size
            test_end = train_end + self.test_window_size

            train_eras = unique_eras[start_idx:train_end]
            test_eras = unique_eras[train_end:test_end]

            splits.append({
                'train_eras': train_eras,
                'test_eras': test_eras,
                'train_start': train_eras[0],
                'train_end': train_eras[-1],
                'test_start': test_eras[0],
                'test_end': test_eras[-1]
            })

            start_idx += self.step_size

        return splits

    def run_walk_forward(self, train_df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str = 'target',
                         model_params: Dict = None) -> pd.DataFrame:
        """
        Запускает walk-forward analysis

        Returns:
            DataFrame с результатами каждого окна
        """
        if model_params is None:
            model_params = {
                'n_estimators': 1000,
                'learning_rate': 0.01,
                'max_depth': 5,
                'num_leaves': 31,
                'colsample_bytree': 0.1
            }

        splits = self.create_walk_forward_splits(train_df['era'].values)
        results = []

        print(f"Running Walk-Forward Analysis: {len(splits)} windows")

        for i, split in enumerate(tqdm(splits)):
            # Prepare data
            train_mask = train_df['era'].isin(split['train_eras'])
            test_mask = train_df['era'].isin(split['test_eras'])

            X_train = train_df.loc[train_mask, feature_cols].values
            y_train = train_df.loc[train_mask, target_col].values
            X_test = train_df.loc[test_mask, feature_cols].values
            y_test = train_df.loc[test_mask, target_col].values
            test_eras = train_df.loc[test_mask, 'era'].values

            # Train model
            model = lgb.LGBMRegressor(**model_params, verbose=-1)
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Calculate per-era correlation
            test_df = pd.DataFrame({
                'prediction': predictions,
                'target': y_test,
                'era': test_eras
            })

            per_era_corr = test_df.groupby('era').apply(
                lambda x: numerai_corr(x[['prediction']], x['target'])
            ).values.flatten()

            # Store results
            results.append({
                'window': i,
                'train_start': split['train_start'],
                'train_end': split['train_end'],
                'test_start': split['test_start'],
                'test_end': split['test_end'],
                'mean_corr': np.mean(per_era_corr),
                'std_corr': np.std(per_era_corr),
                'sharpe': np.mean(per_era_corr) / np.std(per_era_corr) if np.std(per_era_corr) > 0 else 0,
                'min_corr': np.min(per_era_corr),
                'max_corr': np.max(per_era_corr),
                'n_positive_eras': np.sum(per_era_corr > 0),
                'n_negative_eras': np.sum(per_era_corr < 0)
            })

        return pd.DataFrame(results)

    def plot_walk_forward_results(self, results_df: pd.DataFrame):
        """Визуализация результатов walk-forward"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # 1. Mean Correlation over time
        axes[0].plot(results_df['window'], results_df['mean_corr'],
                     marker='o', linewidth=2, markersize=6)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].fill_between(results_df['window'],
                             results_df['mean_corr'] - results_df['std_corr'],
                             results_df['mean_corr'] + results_df['std_corr'],
                             alpha=0.3)
        axes[0].set_title('Walk-Forward: Mean Correlation Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Window #')
        axes[0].set_ylabel('Mean Correlation')
        axes[0].grid(True, alpha=0.3)

        # 2. Sharpe Ratio over time
        axes[1].plot(results_df['window'], results_df['sharpe'],
                     marker='s', linewidth=2, markersize=6, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Target Sharpe=0.5')
        axes[1].set_title('Walk-Forward: Sharpe Ratio Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Window #')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Win Rate (% positive eras)
        total_eras = results_df['n_positive_eras'] + results_df['n_negative_eras']
        win_rate = results_df['n_positive_eras'] / total_eras * 100
        axes[2].bar(results_df['window'], win_rate, color='steelblue', alpha=0.7)
        axes[2].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
        axes[2].set_title('Walk-Forward: Win Rate (% Positive Eras)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Window #')
        axes[2].set_ylabel('Win Rate (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Summary statistics
        print("\n" + "="*60)
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Windows:        {len(results_df)}")
        print(f"Mean Correlation:     {results_df['mean_corr'].mean():.6f} ± {results_df['mean_corr'].std():.6f}")
        print(f"Mean Sharpe:          {results_df['sharpe'].mean():.4f}")
        print(f"Worst Window Corr:    {results_df['mean_corr'].min():.6f}")
        print(f"Best Window Corr:     {results_df['mean_corr'].max():.6f}")
        print(f"Avg Win Rate:         {win_rate.mean():.2f}%")
        print("="*60)


# ============================================================================
# 2. MONTE CARLO SIMULATION
# ============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo Simulation для оценки риска модели
    """

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def simulate_future_performance(self, historical_correlations: np.ndarray,
                                   n_future_eras: int = 52) -> Dict:
        """
        Симулирует будущую производительность на основе исторических данных

        Args:
            historical_correlations: массив per-era корреляций
            n_future_eras: количество эр для симуляции (52 = 1 год)

        Returns:
            Статистика симуляций
        """
        # Подгоняем нормальное распределение к историческим данным
        mu = np.mean(historical_correlations)
        sigma = np.std(historical_correlations)

        print(f"Historical Stats: μ={mu:.6f}, σ={sigma:.6f}")

        # Запускаем симуляции
        simulated_paths = []
        cumulative_returns = []
        final_correlations = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Симулируем n_future_eras корреляций
            simulated_corrs = np.random.normal(mu, sigma, n_future_eras)
            simulated_paths.append(simulated_corrs)

            # Кумулятивная сумма (аналог доходности)
            cumsum = np.cumsum(simulated_corrs)
            cumulative_returns.append(cumsum[-1])

            # Финальная средняя корреляция
            final_correlations.append(np.mean(simulated_corrs))

            # Max drawdown
            running_max = np.maximum.accumulate(cumsum)
            drawdown = running_max - cumsum
            max_drawdowns.append(np.max(drawdown))

        simulated_paths = np.array(simulated_paths)

        # Статистика
        results = {
            'mu': mu,
            'sigma': sigma,
            'simulated_paths': simulated_paths,
            'cumulative_returns': np.array(cumulative_returns),
            'final_correlations': np.array(final_correlations),
            'max_drawdowns': np.array(max_drawdowns),
            'percentiles': {
                'p5': np.percentile(cumulative_returns, 5),
                'p25': np.percentile(cumulative_returns, 25),
                'p50': np.percentile(cumulative_returns, 50),
                'p75': np.percentile(cumulative_returns, 75),
                'p95': np.percentile(cumulative_returns, 95),
            },
            'risk_metrics': {
                'prob_positive': np.mean(np.array(final_correlations) > 0) * 100,
                'prob_sharpe_gt_05': np.mean(
                    np.array(final_correlations) / sigma > 0.5
                ) * 100,
                'expected_max_drawdown': np.mean(max_drawdowns),
            }
        }

        return results

    def plot_monte_carlo(self, results: Dict, n_future_eras: int = 52):
        """Визуализация Monte Carlo симуляций"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Sample paths
        ax1 = fig.add_subplot(gs[0, :])
        sample_paths = results['simulated_paths'][:100]  # Показываем 100 путей
        for path in sample_paths:
            ax1.plot(np.cumsum(path), alpha=0.1, color='blue')

        # Percentiles
        all_cumsum = np.array([np.cumsum(p) for p in results['simulated_paths']])
        p5 = np.percentile(all_cumsum, 5, axis=0)
        p50 = np.percentile(all_cumsum, 50, axis=0)
        p95 = np.percentile(all_cumsum, 95, axis=0)

        ax1.plot(p50, color='red', linewidth=2, label='Median (P50)')
        ax1.fill_between(range(n_future_eras), p5, p95, alpha=0.3, color='red', label='P5-P95 Range')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title(f'Monte Carlo: {self.n_simulations:,} Simulated Paths ({n_future_eras} eras)',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Era')
        ax1.set_ylabel('Cumulative Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribution of final cumulative returns
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(results['cumulative_returns'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(results['percentiles']['p50'], color='red', linestyle='--',
                   linewidth=2, label=f"Median: {results['percentiles']['p50']:.4f}")
        ax2.axvline(results['percentiles']['p5'], color='orange', linestyle='--',
                   linewidth=2, label=f"P5: {results['percentiles']['p5']:.4f}")
        ax2.axvline(results['percentiles']['p95'], color='green', linestyle='--',
                   linewidth=2, label=f"P95: {results['percentiles']['p95']:.4f}")
        ax2.set_title('Distribution of Final Cumulative Correlation', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Cumulative Correlation')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Distribution of max drawdowns
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(results['max_drawdowns'], bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax3.axvline(results['risk_metrics']['expected_max_drawdown'],
                   color='red', linestyle='--', linewidth=2,
                   label=f"Expected: {results['risk_metrics']['expected_max_drawdown']:.4f}")
        ax3.set_title('Distribution of Max Drawdowns', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Max Drawdown')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Risk metrics summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        summary_text = f"""
        MONTE CARLO SIMULATION SUMMARY ({self.n_simulations:,} simulations)
        {'='*70}

        Historical Parameters:
          • Mean (μ):           {results['mu']:.6f}
          • Std Dev (σ):        {results['sigma']:.6f}

        Percentiles of Final Cumulative Correlation:
          • P5  (5th):          {results['percentiles']['p5']:.6f}
          • P25 (25th):         {results['percentiles']['p25']:.6f}
          • P50 (Median):       {results['percentiles']['p50']:.6f}
          • P75 (75th):         {results['percentiles']['p75']:.6f}
          • P95 (95th):         {results['percentiles']['p95']:.6f}

        Risk Metrics:
          • Probability of Positive Performance:  {results['risk_metrics']['prob_positive']:.2f}%
          • Probability of Sharpe > 0.5:          {results['risk_metrics']['prob_sharpe_gt_05']:.2f}%
          • Expected Max Drawdown:                {results['risk_metrics']['expected_max_drawdown']:.6f}

        Interpretation:
          • {results['risk_metrics']['prob_positive']:.1f}% chance of positive mean correlation over next {n_future_eras} eras
          • 95% confidence interval: [{results['percentiles']['p5']:.6f}, {results['percentiles']['p95']:.6f}]
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.show()


# ============================================================================
# 3. NORMAL DISTRIBUTION TESTING
# ============================================================================

class NormalityTester:
    """
    Проверка нормальности распределения корреляций
    """

    @staticmethod
    def test_normality(correlations: np.ndarray) -> Dict:
        """
        Тесты на нормальность распределения

        Returns:
            Результаты статистических тестов
        """
        # 1. Shapiro-Wilk Test
        shapiro_stat, shapiro_p = stats.shapiro(correlations)

        # 2. Kolmogorov-Smirnov Test
        ks_stat, ks_p = stats.kstest(correlations, 'norm',
                                     args=(np.mean(correlations), np.std(correlations)))

        # 3. Anderson-Darling Test
        anderson_result = stats.anderson(correlations, dist='norm')

        # 4. Jarque-Bera Test
        jb_stat, jb_p = stats.jarque_bera(correlations)

        # 5. D'Agostino-Pearson Test
        k2_stat, k2_p = stats.normaltest(correlations)

        # Summary statistics
        results = {
            'mean': np.mean(correlations),
            'std': np.std(correlations),
            'skewness': stats.skew(correlations),
            'kurtosis': stats.kurtosis(correlations),
            'tests': {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p,
                                'is_normal': shapiro_p > 0.05},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p,
                                      'is_normal': ks_p > 0.05},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p,
                               'is_normal': jb_p > 0.05},
                'dagostino_pearson': {'statistic': k2_stat, 'p_value': k2_p,
                                     'is_normal': k2_p > 0.05},
                'anderson_darling': {
                    'statistic': anderson_result.statistic,
                    'critical_values': anderson_result.critical_values,
                    'significance_levels': anderson_result.significance_level
                }
            }
        }

        return results

    @staticmethod
    def plot_normality_tests(correlations: np.ndarray, results: Dict):
        """Визуализация тестов нормальности"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Histogram with normal fit
        ax1 = axes[0, 0]
        ax1.hist(correlations, bins=30, density=True, alpha=0.7,
                color='steelblue', edgecolor='black', label='Empirical')

        # Fit normal distribution
        mu, sigma = results['mean'], results['std']
        x = np.linspace(correlations.min(), correlations.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        ax1.set_title('Histogram vs Normal Distribution', fontweight='bold')
        ax1.set_xlabel('Correlation')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Q-Q Plot
        ax2 = axes[0, 1]
        stats.probplot(correlations, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Quantile-Quantile)', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Box plot
        ax3 = axes[1, 0]
        ax3.boxplot(correlations, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax3.axhline(mu, color='red', linestyle='--', label=f'Mean: {mu:.4f}')
        ax3.axhline(mu + sigma, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mu+sigma:.4f}')
        ax3.axhline(mu - sigma, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mu-sigma:.4f}')
        ax3.set_title('Box Plot with Statistics', fontweight='bold')
        ax3.set_ylabel('Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
        NORMALITY TEST RESULTS
        {'='*50}

        Distribution Parameters:
          Mean (μ):        {results['mean']:.6f}
          Std Dev (σ):     {results['std']:.6f}
          Skewness:        {results['skewness']:.6f}
          Kurtosis:        {results['kurtosis']:.6f}

        Statistical Tests (α=0.05):
          Shapiro-Wilk:    {'NORMAL ✓' if results['tests']['shapiro_wilk']['is_normal'] else 'NOT NORMAL ✗'}
                          (p={results['tests']['shapiro_wilk']['p_value']:.4f})

          Kolmogorov-Smirnov: {'NORMAL ✓' if results['tests']['kolmogorov_smirnov']['is_normal'] else 'NOT NORMAL ✗'}
                          (p={results['tests']['kolmogorov_smirnov']['p_value']:.4f})

          Jarque-Bera:     {'NORMAL ✓' if results['tests']['jarque_bera']['is_normal'] else 'NOT NORMAL ✗'}
                          (p={results['tests']['jarque_bera']['p_value']:.4f})

          D'Agostino:      {'NORMAL ✓' if results['tests']['dagostino_pearson']['is_normal'] else 'NOT NORMAL ✗'}
                          (p={results['tests']['dagostino_pearson']['p_value']:.4f})

        Interpretation:
          {'Data appears normally distributed' if sum([t['is_normal'] for t in results['tests'].values() if isinstance(t, dict) and 'is_normal' in t]) >= 3 else 'Data may not be normally distributed'}
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.show()


# ============================================================================
# 4. MODEL DEPLOYMENT FOR NUMERAI
# ============================================================================

class NumeraiModelDeployer:
    """
    Подготовка и сохранение модели для Numerai
    """

    @staticmethod
    def create_prediction_function(model, feature_cols: List[str],
                                   apply_neutralization: bool = False,
                                   neutralization_features: List[str] = None,
                                   neutralization_proportion: float = 0.5):
        """
        Создает функцию предсказания для Numerai

        Args:
            model: обученная модель
            feature_cols: список признаков
            apply_neutralization: применять ли нейтрализацию
            neutralization_features: признаки для нейтрализации
            neutralization_proportion: степень нейтрализации (0-1)
        """
        def predict(live_features: pd.DataFrame,
                   _live_benchmark_models: pd.DataFrame = None) -> pd.DataFrame:
            """
            Функция предсказания для Numerai Model Upload
            """
            # Generate predictions
            predictions = model.predict(live_features[feature_cols])
            predictions_df = pd.DataFrame(predictions,
                                         index=live_features.index,
                                         columns=['prediction'])

            # Apply neutralization if needed
            if apply_neutralization and neutralization_features:
                from numerai_tools.scoring import neutralize
                predictions_df = neutralize(
                    predictions_df,
                    live_features[neutralization_features],
                    proportion=neutralization_proportion
                )

            # Rank predictions (required by Numerai)
            predictions_df['prediction'] = predictions_df['prediction'].rank(pct=True, method='first')

            return predictions_df

        return predict

    @staticmethod
    def save_model_for_numerai(model, feature_cols: List[str],
                               filename: str = "numerai_model.pkl",
                               apply_neutralization: bool = False,
                               neutralization_features: List[str] = None,
                               neutralization_proportion: float = 0.5):
        """
        Сохраняет модель в формате cloudpickle для Numerai

        Args:
            model: обученная модель
            feature_cols: список признаков
            filename: имя файла для сохранения
            apply_neutralization: применять ли нейтрализацию
            neutralization_features: признаки для нейтрализации
            neutralization_proportion: степень нейтрализации
        """
        # Create prediction function
        predict_fn = NumeraiModelDeployer.create_prediction_function(
            model=model,
            feature_cols=feature_cols,
            apply_neutralization=apply_neutralization,
            neutralization_features=neutralization_features,
            neutralization_proportion=neutralization_proportion
        )

        # Serialize with cloudpickle
        pickled = cloudpickle.dumps(predict_fn)

        # Save to file
        with open(filename, 'wb') as f:
            f.write(pickled)

        print(f"\n{'='*60}")
        print(f"✅ Model saved successfully: {filename}")
        print(f"{'='*60}")
        print(f"Features used: {len(feature_cols)}")
        print(f"Neutralization: {'Enabled' if apply_neutralization else 'Disabled'}")
        if apply_neutralization:
            print(f"  - Features: {len(neutralization_features) if neutralization_features else 0}")
            print(f"  - Proportion: {neutralization_proportion}")
        print(f"{'='*60}")
        print(f"\n📤 Upload this file to Numerai:")
        print(f"   https://numer.ai/models")
        print(f"{'='*60}\n")

        return filename

    @staticmethod
    def test_model_locally(model_file: str, data_version: str = "v5.2"):
        """
        Тестирует сохраненную модель локально перед загрузкой
        """
        print(f"\n{'='*60}")
        print(f"Testing model: {model_file}")
        print(f"{'='*60}\n")

        # Load the pickled function
        with open(model_file, 'rb') as f:
            predict_fn = cloudpickle.load(f)

        # Download live data
        print("Downloading live data...")
        napi = NumerAPI()
        napi.download_dataset(f"{data_version}/live.parquet")

        # Load feature metadata
        feature_metadata = json.load(open(f"{data_version}/features.json"))
        all_features = feature_metadata["feature_sets"]["all"]

        # Load live features
        live_features = pd.read_parquet(f"{data_version}/live.parquet",
                                       columns=all_features)

        print(f"Live features shape: {live_features.shape}")

        # Make predictions
        print("Generating predictions...")
        predictions = predict_fn(live_features)

        # Validate predictions
        print(f"\n{'='*60}")
        print("PREDICTION VALIDATION")
        print(f"{'='*60}")
        print(f"Shape: {predictions.shape}")
        print(f"Min: {predictions['prediction'].min():.6f}")
        print(f"Max: {predictions['prediction'].max():.6f}")
        print(f"Mean: {predictions['prediction'].mean():.6f}")
        print(f"Std: {predictions['prediction'].std():.6f}")
        print(f"NaNs: {predictions['prediction'].isna().sum()}")

        # Check if predictions are properly ranked
        is_ranked = (predictions['prediction'].min() >= 0) and (predictions['prediction'].max() <= 1)
        print(f"\nProperly ranked: {'✅ YES' if is_ranked else '❌ NO'}")

        if not is_ranked:
            print("⚠️ WARNING: Predictions should be ranked between 0 and 1")

        print(f"{'='*60}\n")

        # Display sample predictions
        print("Sample predictions:")
        print(predictions.head(10))

        return predictions


# ============================================================================
# 5. MAIN ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis():
    """
    Полный анализ с Walk-Forward, Monte Carlo и тестами нормальности
    """
    print("="*80)
    print("NUMERAI ADVANCED VALIDATION FRAMEWORK")
    print("="*80)

    # Download data
    print("\n[1/6] Downloading data...")
    napi = NumerAPI()
    DATA_VERSION = "v5.2"

    napi.download_dataset(f"{DATA_VERSION}/train.parquet")
    napi.download_dataset(f"{DATA_VERSION}/features.json")

    # Load data
    print("[2/6] Loading data...")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_cols = feature_metadata["feature_sets"]["small"]

    train = pd.read_parquet(
        f"{DATA_VERSION}/train.parquet",
        columns=["era", "target"] + feature_cols
    )
    train = train[train["era"].isin(train["era"].unique()[::4])]  # Downsample

    print(f"Data shape: {train.shape}")

    # ========================================================================
    # WALK-FORWARD ANALYSIS
    # ========================================================================
    print("\n[3/6] Running Walk-Forward Analysis...")
    wf_analyzer = WalkForwardAnalyzer(
        train_window_size=100,  # ~2 years
        test_window_size=20,    # ~5 months
        step_size=10            # Step forward 10 eras
    )

    wf_results = wf_analyzer.run_walk_forward(
        train_df=train,
        feature_cols=feature_cols,
        target_col='target'
    )

    wf_analyzer.plot_walk_forward_results(wf_results)

    # Collect all per-era correlations for further analysis
    print("\n[4/6] Collecting historical correlations...")
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01,
                             max_depth=5, num_leaves=31, colsample_bytree=0.1,
                             verbose=-1)
    model.fit(train[feature_cols], train['target'])

    train['prediction'] = model.predict(train[feature_cols])
    historical_corrs = train.groupby('era').apply(
        lambda x: numerai_corr(x[['prediction']], x['target'])
    ).values.flatten()

    # ========================================================================
    # NORMALITY TESTING
    # ========================================================================
    print("\n[5/6] Testing Normality of Correlations...")
    normality_tester = NormalityTester()
    normality_results = normality_tester.test_normality(historical_corrs)
    normality_tester.plot_normality_tests(historical_corrs, normality_results)

    # ========================================================================
    # MONTE CARLO SIMULATION
    # ========================================================================
    print("\n[6/6] Running Monte Carlo Simulation...")
    mc_simulator = MonteCarloSimulator(n_simulations=10000)
    mc_results = mc_simulator.simulate_future_performance(
        historical_correlations=historical_corrs,
        n_future_eras=52  # 1 year ahead
    )
    mc_simulator.plot_monte_carlo(mc_results, n_future_eras=52)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    # ========================================================================
    # SAVE MODEL FOR NUMERAI
    # ========================================================================
    print("\n[BONUS] Saving model for Numerai deployment...")

    # Train final model on all data
    print("Training final model on all available data...")
    final_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.1,
        verbose=-1
    )
    final_model.fit(train[feature_cols], train['target'])

    # Save model
    model_filename = NumeraiModelDeployer.save_model_for_numerai(
        model=final_model,
        feature_cols=feature_cols,
        filename="numerai_validated_model.pkl",
        apply_neutralization=False,  # Set to True if you want neutralization
        neutralization_features=None,  # Or specify features to neutralize
        neutralization_proportion=0.5
    )

    # Test model locally
    print("\nTesting saved model...")
    test_predictions = NumeraiModelDeployer.test_model_locally(model_filename)

    return {
        'walk_forward': wf_results,
        'normality': normality_results,
        'monte_carlo': mc_results,
        'historical_correlations': historical_corrs,
        'final_model': final_model,
        'model_file': model_filename,
        'test_predictions': test_predictions
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def quick_model_validation(feature_set: str = "small",
                          save_model: bool = True):
    """
    Быстрая валидация и сохранение модели

    Args:
        feature_set: "small", "medium", or "all"
        save_model: сохранить ли модель для Numerai
    """
    print("="*80)
    print(f"QUICK MODEL VALIDATION ({feature_set.upper()} features)")
    print("="*80)

    # Setup
    napi = NumerAPI()
    DATA_VERSION = "v5.2"

    # Download data
    print("\n[1/3] Downloading data...")
    napi.download_dataset(f"{DATA_VERSION}/train.parquet")
    napi.download_dataset(f"{DATA_VERSION}/features.json")

    # Load data
    print("[2/3] Loading data...")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_cols = feature_metadata["feature_sets"][feature_set]

    train = pd.read_parquet(
        f"{DATA_VERSION}/train.parquet",
        columns=["era", "target"] + feature_cols
    )
    train = train[train["era"].isin(train["era"].unique()[::4])]

    print(f"Training data: {train.shape}")
    print(f"Features: {len(feature_cols)}")

    # Train model
    print("\n[3/3] Training model...")
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.1,
        verbose=-1
    )
    model.fit(train[feature_cols], train['target'])

    # Quick validation
    print("\nGenerating predictions...")
    train['prediction'] = model.predict(train[feature_cols])

    per_era_corr = train.groupby('era').apply(
        lambda x: numerai_corr(x[['prediction']], x['target'])
    ).values.flatten()

    print(f"\n{'='*60}")
    print("QUICK VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Correlation: {np.mean(per_era_corr):.6f}")
    print(f"Std Correlation:  {np.std(per_era_corr):.6f}")
    print(f"Sharpe Ratio:     {np.mean(per_era_corr) / np.std(per_era_corr):.4f}")
    print(f"Min Correlation:  {np.min(per_era_corr):.6f}")
    print(f"Max Correlation:  {np.max(per_era_corr):.6f}")
    print(f"% Positive Eras:  {(per_era_corr > 0).sum() / len(per_era_corr) * 100:.1f}%")
    print(f"{'='*60}\n")

    if save_model:
        # Save model
        model_filename = NumeraiModelDeployer.save_model_for_numerai(
            model=model,
            feature_cols=feature_cols,
            filename=f"numerai_model_monte_{feature_set}.pkl"
        )

        # Test model
        NumeraiModelDeployer.test_model_locally(model_filename, DATA_VERSION)

        return model, model_filename

    return model




# Опция 2: Быстрая валидация и сохранение модели
#model, model_file = quick_model_validation(feature_set="small", save_model=True)
results = run_complete_analysis()
