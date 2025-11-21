"""
XGBoost-Based Regression Model for Score Prediction
Utilizes E5-Large embeddings with comprehensive feature engineering
Target Performance: RMSE 2.5-3.5
Approximate Runtime: 1 hour
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
import xgboost as xgb


class SystemConfiguration:
    """Global configuration parameters"""
    RANDOM_SEED = 42
    EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
    ENCODING_BATCH_SIZE = 128
    CV_SPLITS = 5
    XGBOOST_ROUNDS = 1000
    EARLY_STOP_ROUNDS = 50
    VERBOSE_FREQUENCY = 20


def initialize_reproducibility(seed_val):
    """Ensure reproducible results across runs"""
    np.random.seed(seed_val)


class EmbeddingProcessor:
    """Manages text encoding using transformer models"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.transformer_model = None
    
    def initialize_model(self):
        """Load the sentence transformer"""
        print(f"Initializing transformer model: {self.model_path}")
        self.transformer_model = SentenceTransformer(self.model_path)
        return self
    
    def construct_text_representation(self, data_record):
        """Build formatted text from conversation components"""
        user_input = data_record['user_prompt']
        system_context = data_record.get('system_prompt', '')
        model_output = data_record['response']
        
        # Check if E5 model requires special formatting
        uses_e5_format = 'e5' in self.model_path.lower()
        
        if uses_e5_format:
            formatted_text = (
                f"passage: Prompt: {user_input} "
                f"System: {system_context} "
                f"Response: {model_output}"
            )
        else:
            formatted_text = (
                f"Prompt: {user_input} "
                f"System: {system_context} "
                f"Response: {model_output}"
            )
        
        return formatted_text
    
    def encode_data_collection(self, data_collection, batch_size):
        """Transform text data into vector embeddings"""
        text_representations = [
            self.construct_text_representation(record) 
            for record in data_collection
        ]
        
        embeddings = self.transformer_model.encode(
            text_representations,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def process_datasets(self, training_data, testing_data):
        """Generate embeddings for both training and test sets"""
        config = SystemConfiguration()
        
        print("Encoding training dataset...")
        train_vectors = self.encode_data_collection(
            training_data, 
            config.ENCODING_BATCH_SIZE
        )
        
        print("Encoding test dataset...")
        test_vectors = self.encode_data_collection(
            testing_data, 
            config.ENCODING_BATCH_SIZE
        )
        
        return train_vectors, test_vectors


class FeatureEngineer:
    """Creates comprehensive feature sets from embeddings"""
    
    @staticmethod
    def compute_similarity_metrics(vector_a, vector_b):
        """Calculate various distance and similarity measures with padding."""

        # --- Fix: Pad vectors to same dimension ---
        len_a, len_b = len(vector_a), len(vector_b)
        if len_a != len_b:
            max_len = max(len_a, len_b)
            vector_a = np.pad(vector_a, (0, max_len - len_a), mode="constant")
            vector_b = np.pad(vector_b, (0, max_len - len_b), mode="constant")

        metrics = {}

        # Dot product similarity
        metrics['dot_product'] = np.dot(vector_a, vector_b)

        # Euclidean distance
        metrics['l2_distance'] = np.linalg.norm(vector_a - vector_b)

        # Manhattan distance
        metrics['l1_distance'] = np.linalg.norm(vector_a - vector_b, ord=1)

        # Cosine similarity (safe)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            metrics['cosine_similarity'] = 0.0
        else:
            metrics['cosine_similarity'] = metrics['dot_product'] / (norm_a * norm_b)

        return metrics
    
    @staticmethod
    def compute_statistical_features(difference_vector):
        """Extract statistical properties from difference vector"""
        stats = {}
        
        stats['mean'] = np.mean(difference_vector)
        stats['std'] = np.std(difference_vector)
        stats['median'] = np.median(difference_vector)
        stats['max'] = np.max(difference_vector)
        stats['min'] = np.min(difference_vector)
        stats['q25'] = np.percentile(difference_vector, 25)
        stats['q75'] = np.percentile(difference_vector, 75)
        
        return stats
    
    def build_feature_vector(self, text_embedding, metric_embedding):
        """Construct comprehensive feature vector from embedding pair"""
        feature_list = []
        
        # Core embeddings concatenation
        feature_list.extend(text_embedding)
        feature_list.extend(metric_embedding)
        
        # Handle mismatch by padding
        text_dim = len(text_embedding)
        metric_dim = len(metric_embedding)
        
        if text_dim != metric_dim:
            max_dim = max(text_dim, metric_dim)
            text_padded = np.pad(text_embedding, (0, max_dim - text_dim), mode='constant')
            metric_padded = np.pad(metric_embedding, (0, max_dim - metric_dim), mode='constant')
        else:
            text_padded = text_embedding
            metric_padded = metric_embedding
        
        # Element-wise operations
        feature_list.extend(text_padded * metric_padded)  
        feature_list.extend(text_padded + metric_padded)  
        feature_list.extend(np.abs(text_padded - metric_padded))
        
        # Corrected similarity
        sim = self.compute_similarity_metrics(text_embedding, metric_embedding)
        feature_list.append(sim['dot_product'])
        feature_list.append(sim['l2_distance'])
        feature_list.append(sim['l1_distance'])
        feature_list.append(sim['cosine_similarity'])
        
        # Statistical features
        diff = text_padded - metric_padded
        stats = self.compute_statistical_features(diff)
        feature_list.append(stats['mean'])
        feature_list.append(stats['std'])
        feature_list.append(stats['median'])
        feature_list.append(stats['max'])
        feature_list.append(stats['min'])
        feature_list.append(stats['q25'])
        feature_list.append(stats['q75'])
        
        return feature_list
    
    def transform_embeddings_to_features(self, text_embeddings, 
                                         metric_embeddings, metric_idx_list):
        """Convert embedding matrices to feature matrix"""
        num_samples = len(text_embeddings)
        all_features = []
        
        print("Engineering features...")
        for i in tqdm(range(num_samples)):
            text_vec = text_embeddings[i]
            metric_vec = metric_embeddings[metric_idx_list[i]]
            
            fv = self.build_feature_vector(text_vec, metric_vec)
            all_features.append(fv)
        
        return np.array(all_features)


class XGBoostTrainer:
    """Handles XGBoost model training and evaluation"""
    
    def __init__(self):
        self.config = SystemConfiguration()
        self.hyperparameters = self._define_hyperparameters()
    
    def _define_hyperparameters(self):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 1,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': self.config.RANDOM_SEED,
            'tree_method': 'hist',
            'device': 'cuda'
        }
        return params
    
    def train_model(self, features_train, targets_train, 
                   features_val, targets_val):
        
        dtrain = xgb.DMatrix(features_train, label=targets_train)
        dval = xgb.DMatrix(features_val, label=targets_val)
        
        eval_list = [(dtrain, 'train'), (dval, 'val')]
        
        model = xgb.train(
            self.hyperparameters,
            dtrain,
            num_boost_round=self.config.XGBOOST_ROUNDS,
            evals=eval_list,
            early_stopping_rounds=self.config.EARLY_STOP_ROUNDS,
            verbose_eval=self.config.VERBOSE_FREQUENCY
        )
        
        preds = model.predict(dval)
        rmse = np.sqrt(np.mean((preds - targets_val) ** 2))
        
        print(f"Optimal iteration: {model.best_iteration}")
        print(f"Validation RMSE: {rmse:.4f}")
        
        return model, rmse
    
    def generate_predictions(self, model, feature_matrix):
        dmatrix = xgb.DMatrix(feature_matrix)
        return model.predict(dmatrix)


class FileManager:
    """Handles file I/O operations"""
    
    @staticmethod
    def read_json_data(path):
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def read_numpy_array(path):
        return np.load(path)
    
    @staticmethod
    def write_numpy_array(arr, path):
        np.save(path, arr)
    
    @staticmethod
    def write_csv(df, path):
        df.to_csv(path, index=False)


class CrossValidationManager:
    """Orchestrates k-fold cross-validation"""
    
    def __init__(self):
        self.config = SystemConfiguration()
        self.validation_scores = []
        self.fold_predictions = []
        self.trained_models = []
    
    def execute_fold(self, fold, train_idx, val_idx,
                     feature_matrix, target_vector, test_features):
        
        print(f"\n========== FOLD {fold+1}/{self.config.CV_SPLITS} ==========")
        
        X_train = feature_matrix[train_idx]
        y_train = target_vector[train_idx]
        X_val = feature_matrix[val_idx]
        y_val = target_vector[val_idx]
        
        trainer = XGBoostTrainer()
        model, rmse = trainer.train_model(
            X_train, y_train,
            X_val, y_val
        )
        
        preds = trainer.generate_predictions(model, test_features)
        preds = np.clip(preds, 0, 10)
        
        return model, rmse, preds
    
    def run_cross_validation(self, feature_matrix, target_vector, test_features):
        
        bins = np.round(target_vector).astype(int)
        
        kf = StratifiedKFold(
            n_splits=self.config.CV_SPLITS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(feature_matrix, bins)):
            
            model, rmse, preds = self.execute_fold(
                fold_idx, train_idx, val_idx,
                feature_matrix, target_vector, test_features
            )
            
            self.validation_scores.append(rmse)
            self.fold_predictions.append(preds)
            self.trained_models.append(model)
        
        return self.validation_scores, self.fold_predictions, self.trained_models


class FeatureImportanceAnalyzer:
    """Feature importance reporting"""
    
    @staticmethod
    def extract_top_features(model, top_n=20):
        imp = model.get_score(importance_type='gain')
        return sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    @staticmethod
    def display_feature_importance(model):
        print("\nTop 20 Important Features:")
        for k, v in FeatureImportanceAnalyzer.extract_top_features(model):
            print(f"{k}: {v:.2f}")


def main():
    config = SystemConfiguration()
    initialize_reproducibility(config.RANDOM_SEED)
    
    file_mgr = FileManager()
    
    print("Loading datasets...")
    train_dataset = file_mgr.read_json_data('/content/sample_data/train_data.json')
    test_dataset = file_mgr.read_json_data('/content/sample_data/test_data.json')
    metric_names_list = file_mgr.read_json_data('/content/sample_data/metric_names.json')
    metric_embeddings = file_mgr.read_numpy_array('/content/sample_data/metric_name_embeddings.npy')
    
    metric_name_to_index = {name: idx for idx, name in enumerate(metric_names_list)}
    
    try:
        print("Loading cached embeddings...")
        train_text = file_mgr.read_numpy_array('train_text_embeddings.npy')
        test_text = file_mgr.read_numpy_array('test_text_embeddings.npy')
    except FileNotFoundError:
        print("Generating new embeddings...")
        ep = EmbeddingProcessor(config.EMBEDDING_MODEL).initialize_model()
        train_text, test_text = ep.process_datasets(train_dataset, test_dataset)
        file_mgr.write_numpy_array(train_text, 'train_text_embeddings.npy')
        file_mgr.write_numpy_array(test_text, 'test_text_embeddings.npy')
    
    print(f"Embedding dimensions: train={train_text.shape}, test={test_text.shape}")
    
    train_metric_idx = [metric_name_to_index[x["metric_name"]] for x in train_dataset]
    test_metric_idx = [metric_name_to_index[x["metric_name"]] for x in test_dataset]
    target_scores = np.array([float(x["score"]) for x in train_dataset])
    
    engineer = FeatureEngineer()
    
    print("\nEngineering training features...")
    train_features = engineer.transform_embeddings_to_features(
        train_text, metric_embeddings, train_metric_idx
    )
    
    print("Engineering test features...")
    test_features = engineer.transform_embeddings_to_features(
        test_text, metric_embeddings, test_metric_idx
    )
    
    print(f"Final feature matrix shape: {train_features.shape}")
    
    cv = CrossValidationManager()
    cv_scores, fold_preds, models = cv.run_cross_validation(
        train_features, target_scores, test_features
    )
    
    print("\n=========== CV SUMMARY ===========")
    print(f"RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    ensemble = np.mean(fold_preds, axis=0)
    ensemble = np.clip(ensemble, 0, 10)
    ensemble = np.round(ensemble)
    
    submission = pd.DataFrame({
        "ID": range(1, len(ensemble) + 1),
        "score": ensemble
    })
    file_mgr.write_csv(submission, 'submission_xgboost.csv')
    
    print("\nSubmission saved as submission_xgboost.csv")
    print(f"Prediction stats: mean={ensemble.mean():.2f}, std={ensemble.std():.2f}")
    
    FeatureImportanceAnalyzer.display_feature_importance(models[0])


if __name__ == '__main__':
    main()
