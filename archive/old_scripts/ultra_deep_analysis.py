"""
Ultra-Deep Data Analysis
超深度數據分析 - 找出隱藏的模式和洞見
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter, defaultdict
from PIL import Image
import os
from tqdm import tqdm

class UltraDeepAnalyzer:
    """Ultra Deep Data Analyzer"""

    def __init__(self, data_dir='data', train_images='train_images', val_images='val_images', test_images='test_images'):
        self.data_dir = Path(data_dir)
        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images
        self.report = {}

    def analyze_all(self):
        """Run all analyses"""
        print("🔬 Ultra-Deep Data Analysis Starting...")
        print("=" * 80)

        # 1. CSV Structure Analysis
        print("\n📊 [1/10] Analyzing CSV structures...")
        self.analyze_csv_structures()

        # 2. Class Distribution Deep Dive
        print("\n📈 [2/10] Deep diving into class distributions...")
        self.analyze_class_distributions()

        # 3. Image Statistics
        print("\n🖼️  [3/10] Analyzing image statistics...")
        self.analyze_image_statistics()

        # 4. Filename Patterns
        print("\n📝 [4/10] Detecting filename patterns...")
        self.analyze_filename_patterns()

        # 5. Cross-validation Fold Analysis
        print("\n🔄 [5/10] Analyzing K-Fold splits...")
        self.analyze_kfold_splits()

        # 6. Submission Predictions Analysis
        print("\n🎯 [6/10] Analyzing submission predictions...")
        self.analyze_submissions()

        # 7. Data Leakage Detection
        print("\n🔍 [7/10] Checking for data leakage...")
        self.check_data_leakage()

        # 8. Outlier Detection
        print("\n⚠️  [8/10] Detecting outliers...")
        self.detect_outliers()

        # 9. Missing Data Analysis
        print("\n❓ [9/10] Analyzing missing data...")
        self.analyze_missing_data()

        # 10. Generate Insights
        print("\n💡 [10/10] Generating actionable insights...")
        self.generate_insights()

        # Save report
        self.save_report()

        print("\n" + "=" * 80)
        print("✅ Ultra-Deep Analysis Complete!")

    def analyze_csv_structures(self):
        """Analyze CSV file structures"""
        csv_files = list(self.data_dir.glob('*.csv'))
        structures = {}

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                structures[csv_file.name] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'null_counts': df.isnull().sum().to_dict()
                }
                print(f"  ✓ {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"  ✗ Error reading {csv_file.name}: {e}")

        self.report['csv_structures'] = structures

    def analyze_class_distributions(self):
        """Deep analysis of class distributions"""
        distributions = {}

        # Analyze train/val/test
        for name, csv_file in [('train', 'train_data.csv'), ('val', 'val_data.csv'), ('test', 'test_data.csv')]:
            csv_path = self.data_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)

                # Get class columns
                class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
                if all(col in df.columns for col in class_cols):
                    # Count each class
                    class_counts = {}
                    for col in class_cols:
                        count = (df[col] == 1.0).sum() if name != 'test' else 0
                        class_counts[col] = int(count)

                    distributions[name] = {
                        'total_samples': len(df),
                        'class_counts': class_counts
                    }

                    if name != 'test':
                        # Calculate imbalance ratios
                        counts = list(class_counts.values())
                        max_count = max(counts)
                        min_count = min([c for c in counts if c > 0])
                        distributions[name]['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')

                    print(f"  {name:8s}: {len(df):4d} samples")
                    if name != 'test':
                        for cls, cnt in class_counts.items():
                            pct = cnt / len(df) * 100 if len(df) > 0 else 0
                            print(f"    {cls:12s}: {cnt:4d} ({pct:5.2f}%)")

        self.report['class_distributions'] = distributions

    def analyze_image_statistics(self):
        """Analyze image properties"""
        stats = {}

        for split_name, img_dir in [('train', self.train_images), ('val', self.val_images), ('test', self.test_images)]:
            if not os.path.exists(img_dir):
                continue

            image_files = list(Path(img_dir).glob('*.*'))[:100]  # Sample 100 images

            widths, heights, aspects = [], [], []
            file_sizes = []

            for img_path in tqdm(image_files, desc=f"Sampling {split_name} images", leave=False):
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        aspects.append(w / h)
                    file_sizes.append(img_path.stat().st_size / 1024)  # KB
                except:
                    pass

            if widths:
                stats[split_name] = {
                    'width': {'mean': np.mean(widths), 'std': np.std(widths), 'min': np.min(widths), 'max': np.max(widths)},
                    'height': {'mean': np.mean(heights), 'std': np.std(heights), 'min': np.min(heights), 'max': np.max(heights)},
                    'aspect_ratio': {'mean': np.mean(aspects), 'std': np.std(aspects)},
                    'file_size_kb': {'mean': np.mean(file_sizes), 'std': np.std(file_sizes)}
                }
                print(f"  {split_name:8s}: {len(widths)} images sampled")
                print(f"    Dimensions: {int(stats[split_name]['width']['mean'])}x{int(stats[split_name]['height']['mean'])} ±{int(stats[split_name]['width']['std'])}")

        self.report['image_statistics'] = stats

    def analyze_filename_patterns(self):
        """Analyze filename patterns"""
        patterns = {}

        for name, csv_file in [('train', 'train_data.csv'), ('val', 'val_data.csv'), ('test', 'test_data.csv')]:
            csv_path = self.data_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                file_col = 'new_filename' if 'new_filename' in df.columns else 'filename'

                filenames = df[file_col].tolist()

                # Extract extensions
                extensions = Counter([Path(f).suffix for f in filenames])

                # Check if numeric IDs
                numeric_ids = []
                for f in filenames:
                    stem = Path(f).stem
                    if stem.isdigit():
                        numeric_ids.append(int(stem))

                patterns[name] = {
                    'extensions': dict(extensions),
                    'numeric_ids': len(numeric_ids) > 0,
                    'id_range': (min(numeric_ids), max(numeric_ids)) if numeric_ids else None
                }

                print(f"  {name:8s}: {dict(extensions)}, Numeric IDs: {patterns[name]['numeric_ids']}")

        self.report['filename_patterns'] = patterns

    def analyze_kfold_splits(self):
        """Analyze K-Fold split quality"""
        fold_analysis = {}

        for fold_id in range(5):
            train_csv = self.data_dir / f'fold{fold_id}_train.csv'
            val_csv = self.data_dir / f'fold{fold_id}_val.csv'

            if train_csv.exists() and val_csv.exists():
                train_df = pd.read_csv(train_csv)
                val_df = pd.read_csv(val_csv)

                # Check class distribution
                class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
                train_dist = {col: (train_df[col] == 1.0).sum() for col in class_cols}
                val_dist = {col: (val_df[col] == 1.0).sum() for col in class_cols}

                fold_analysis[f'fold{fold_id}'] = {
                    'train_samples': len(train_df),
                    'val_samples': len(val_df),
                    'train_distribution': train_dist,
                    'val_distribution': val_dist
                }

                print(f"  Fold {fold_id}: Train={len(train_df)}, Val={len(val_df)}, COVID-19 val={val_dist['COVID-19']}")

        self.report['kfold_analysis'] = fold_analysis

    def analyze_submissions(self):
        """Analyze submission predictions"""
        submissions = {}

        submission_files = list(self.data_dir.glob('submission*.csv'))

        for sub_file in submission_files:
            try:
                df = pd.read_csv(sub_file)
                class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

                # Get predictions
                if all(col in df.columns for col in class_cols):
                    # Probability based
                    pred_classes = df[class_cols].values.argmax(axis=1)
                    pred_dist = Counter([class_cols[i] for i in pred_classes])

                    # Check confidence
                    max_probs = df[class_cols].values.max(axis=1)
                    avg_confidence = max_probs.mean()

                    submissions[sub_file.name] = {
                        'distribution': dict(pred_dist),
                        'avg_confidence': float(avg_confidence)
                    }
            except:
                pass

        self.report['submissions'] = submissions
        print(f"  Analyzed {len(submissions)} submission files")

    def check_data_leakage(self):
        """Check for potential data leakage"""
        leakage_report = {}

        # Check if train/val/test have overlapping files
        train_csv = self.data_dir / 'train_data.csv'
        val_csv = self.data_dir / 'val_data.csv'
        test_csv = self.data_dir / 'test_data.csv'

        if all(f.exists() for f in [train_csv, val_csv, test_csv]):
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)

            file_col = 'new_filename' if 'new_filename' in train_df.columns else 'filename'

            train_files = set(train_df[file_col])
            val_files = set(val_df[file_col])
            test_files = set(test_df[file_col])

            train_val_overlap = train_files & val_files
            train_test_overlap = train_files & test_files
            val_test_overlap = val_files & test_files

            leakage_report = {
                'train_val_overlap': len(train_val_overlap),
                'train_test_overlap': len(train_test_overlap),
                'val_test_overlap': len(val_test_overlap)
            }

            print(f"  Train-Val overlap: {len(train_val_overlap)}")
            print(f"  Train-Test overlap: {len(train_test_overlap)}")
            print(f"  Val-Test overlap: {len(val_test_overlap)}")

        self.report['data_leakage'] = leakage_report

    def detect_outliers(self):
        """Detect potential outliers"""
        outliers = {}

        # Check image sizes
        for split_name, img_dir in [('train', self.train_images), ('val', self.val_images)]:
            if not os.path.exists(img_dir):
                continue

            sizes = []
            files = []

            for img_path in list(Path(img_dir).glob('*.*'))[:500]:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        sizes.append(w * h)
                        files.append(img_path.name)
                except:
                    pass

            if sizes:
                # Detect outliers using IQR
                q1, q3 = np.percentile(sizes, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr

                outlier_indices = [i for i, s in enumerate(sizes) if s < lower_bound or s > upper_bound]

                outliers[split_name] = {
                    'count': len(outlier_indices),
                    'examples': [files[i] for i in outlier_indices[:5]]
                }

                print(f"  {split_name:8s}: {len(outlier_indices)} outliers detected")

        self.report['outliers'] = outliers

    def analyze_missing_data(self):
        """Analyze missing data"""
        missing = {}

        for name, csv_file in [('train', 'train_data.csv'), ('val', 'val_data.csv'), ('test', 'test_data.csv')]:
            csv_path = self.data_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                null_counts = df.isnull().sum()
                missing[name] = {col: int(count) for col, count in null_counts.items() if count > 0}

                if missing[name]:
                    print(f"  {name:8s}: Missing values found!")
                    for col, count in missing[name].items():
                        print(f"    {col}: {count}")
                else:
                    print(f"  {name:8s}: No missing values ✓")

        self.report['missing_data'] = missing

    def generate_insights(self):
        """Generate actionable insights"""
        insights = []

        # Insight 1: Class imbalance
        if 'class_distributions' in self.report:
            train_dist = self.report['class_distributions'].get('train', {})
            if 'imbalance_ratio' in train_dist:
                ratio = train_dist['imbalance_ratio']
                if ratio > 40:
                    insights.append({
                        'type': 'CRITICAL',
                        'category': 'Class Imbalance',
                        'finding': f'Extreme class imbalance detected (ratio: {ratio:.1f}:1)',
                        'recommendation': 'Use strong class weighting (15-20x for minority class), Focal Loss with high gamma (3.5+), or oversampling'
                    })

        # Insight 2: K-Fold validation
        if 'kfold_analysis' in self.report:
            covid_counts = []
            for fold_data in self.report['kfold_analysis'].values():
                covid_counts.append(fold_data['val_distribution']['COVID-19'])

            if min(covid_counts) < 5:
                insights.append({
                    'type': 'WARNING',
                    'category': 'K-Fold Splits',
                    'finding': f'Some folds have very few COVID-19 samples (min: {min(covid_counts)})',
                    'recommendation': 'Consider stratified sampling or use original train/val split'
                })

        # Insight 3: Model diversity
        if 'submissions' in self.report:
            distributions = [sub['distribution'] for sub in self.report['submissions'].values()]
            if len(distributions) > 1:
                # Check diversity
                covid_preds = [d.get('COVID-19', 0) for d in distributions]
                covid_std = np.std(covid_preds)

                if covid_std > 5:
                    insights.append({
                        'type': 'INFO',
                        'category': 'Model Diversity',
                        'finding': f'High variance in COVID-19 predictions across models (std: {covid_std:.1f})',
                        'recommendation': 'Good model diversity for ensemble. Consider weighted ensemble based on validation performance.'
                    })

        # Insight 4: Submission confidence
        if 'submissions' in self.report:
            confidences = [sub['avg_confidence'] for sub in self.report['submissions'].values()]
            avg_conf = np.mean(confidences)

            if avg_conf < 0.7:
                insights.append({
                    'type': 'WARNING',
                    'category': 'Prediction Confidence',
                    'finding': f'Low average prediction confidence ({avg_conf:.3f})',
                    'recommendation': 'Models are uncertain. Consider: 1) More training epochs, 2) Better augmentation, 3) Larger models'
                })
            elif avg_conf > 0.95:
                insights.append({
                    'type': 'WARNING',
                    'category': 'Prediction Confidence',
                    'finding': f'Very high prediction confidence ({avg_conf:.3f}) - possible overfitting',
                    'recommendation': 'Add more regularization, stronger augmentation, or use label smoothing'
                })

        self.report['insights'] = insights

        print(f"\n  Generated {len(insights)} actionable insights:")
        for i, insight in enumerate(insights, 1):
            print(f"\n  [{insight['type']}] {insight['category']}")
            print(f"    Finding: {insight['finding']}")
            print(f"    Action: {insight['recommendation']}")

    def save_report(self):
        """Save comprehensive report"""
        output_file = 'data/ultra_deep_analysis_report.json'
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        print(f"\n📄 Full report saved to: {output_file}")


if __name__ == '__main__':
    analyzer = UltraDeepAnalyzer()
    analyzer.analyze_all()
