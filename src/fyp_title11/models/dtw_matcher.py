"""
DYNAMIC TIME WARPING (DTW) MATCHER
FYP Requirement: Traditional Matching Algorithm
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class DTWMatcher:
    """
    Dynamic Time Warping for sequence alignment and matching.
    Traditional approach for comparing time-series data.
    """
    
    def __init__(self, 
                 window_size: Optional[int] = None,
                 distance_metric: str = 'euclidean'):
        
        self.window_size = window_size
        self.distance_metric = distance_metric
        
        # Database of sequences
        self.database = {}
        self.sequence_names = []
        self.sequences = []
    
    def add_to_database(self, name: str, sequence: np.ndarray):
        """Add sequence to database"""
        self.database[name] = sequence
        self.sequence_names.append(name)
        self.sequences.append(sequence)
    
    def dtw_distance(self, 
                    seq1: np.ndarray, 
                    seq2: np.ndarray,
                    window: Optional[int] = None) -> float:
        """
        Compute DTW distance between two sequences
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            window: Sakoe-Chiba band window size
            
        Returns:
            DTW distance
        """
        if window is None:
            window = self.window_size
        
        # Use fastdtw for efficient computation
        distance, _ = fastdtw(seq1, seq2, radius=window)
        
        # Normalize by sequence lengths
        normalized_distance = distance / (len(seq1) + len(seq2))
        
        return normalized_distance
    
    def match_sequence(self, 
                      query: np.ndarray, 
                      top_k: int = 5) -> List[Dict]:
        """
        Match query sequence against database
        
        Args:
            query: Query sequence to match
            top_k: Number of top matches to return
            
        Returns:
            List of match results
        """
        if not self.sequences:
            raise ValueError("Database is empty. Add sequences first.")
        
        distances = []
        
        for name, sequence in self.database.items():
            # Compute DTW distance
            distance = self.dtw_distance(query, sequence)
            distances.append({
                'name': name,
                'distance': distance,
                'sequence': sequence
            })
        
        # Sort by distance (lower is better)
        distances.sort(key=lambda x: x['distance'])
        
        # Convert distance to similarity score (0 to 1)
        for result in distances:
            # Normalize similarity: 1 / (1 + distance)
            result['similarity'] = 1.0 / (1.0 + result['distance'])
            result['confidence'] = np.exp(-result['distance'])
        
        return distances[:top_k]
    
    def align_sequences(self, 
                       seq1: np.ndarray, 
                       seq2: np.ndarray) -> Dict:
        """
        Align two sequences and return warping path
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Alignment information
        """
        distance, path = fastdtw(seq1, seq2, radius=self.window_size)
        
        # Extract warping path
        warping_path = np.array(path)
        
        # Compute alignment quality metrics
        alignment_error = self._compute_alignment_error(seq1, seq2, warping_path)
        
        return {
            'distance': distance,
            'warping_path': warping_path,
            'alignment_error': alignment_error,
            'path_length': len(warping_path)
        }
    
    def _compute_alignment_error(self,
                                seq1: np.ndarray,
                                seq2: np.ndarray,
                                path: np.ndarray) -> float:
        """Compute average alignment error"""
        errors = []
        for i, j in path:
            if i < len(seq1) and j < len(seq2):
                error = euclidean(seq1[i], seq2[j])
                errors.append(error)
        
        return np.mean(errors) if errors else 0.0
    
    def batch_match(self, 
                   queries: List[np.ndarray],
                   query_names: Optional[List[str]] = None) -> Dict:
        """
        Match multiple queries against database
        
        Args:
            queries: List of query sequences
            query_names: Names for queries
            
        Returns:
            Batch matching results
        """
        if query_names is None:
            query_names = [f"query_{i}" for i in range(len(queries))]
        
        results = {}
        
        for query, name in zip(queries, query_names):
            matches = self.match_sequence(query)
            results[name] = {
                'query_name': name,
                'query_length': len(query),
                'matches': matches
            }
        
        return results
    
    def save_database(self, filepath: str):
        """Save database to file"""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'database': self.database,
                'window_size': self.window_size
            }, f)
    
    def load_database(self, filepath: str):
        """Load database from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.database = data['database']
        self.window_size = data.get('window_size', None)
        self.sequence_names = list(self.database.keys())
        self.sequences = list(self.database.values())

class EnhancedDTW:
    """Enhanced DTW with multiple features"""
    
    @staticmethod
    def multi_feature_dtw(features1: Dict, 
                         features2: Dict,
                         weights: Optional[Dict] = None) -> float:
        """
        DTW with multiple feature types
        
        Args:
            features1: Dictionary of feature arrays
            features2: Dictionary of feature arrays
            weights: Weights for each feature type
            
        Returns:
            Combined DTW distance
        """
        if weights is None:
            weights = {
                'mfcc': 0.4,
                'chroma': 0.3,
                'spectral': 0.2,
                'rhythm': 0.1
            }
        
        total_distance = 0.0
        
        for feature_name in features1.keys():
            if feature_name in features2:
                # Extract feature arrays
                feat1 = features1[feature_name]
                feat2 = features2[feature_name]
                
                # Ensure same shape
                if feat1.shape[1] != feat2.shape[1]:
                    # Pad or truncate
                    min_dim = min(feat1.shape[1], feat2.shape[1])
                    feat1 = feat1[:, :min_dim]
                    feat2 = feat2[:, :min_dim]
                
                # Compute DTW
                distance, _ = fastdtw(feat1.T, feat2.T)
                
                # Weighted sum
                weight = weights.get(feature_name, 0.1)
                total_distance += weight * distance
        
        return total_distance