"""
================================================================================
SO SÁNH TẤT CẢ PHƯƠNG PHÁP
================================================================================

Script này so sánh kết quả của tất cả các phương pháp tạo rel_feature:
1. Method 1: Global Average (Original V1 - Problematic)
2. Method 2: Caption Matching (V2 - Improved)
3. Method 3: TransE + Weighted Aggregation (SOTA)
4. Method 4: Object Detection + Scene Graph (Full SOTA)

Metrics:
- Unique embedding ratio
- Feature statistics (mean, std, variance)
- Correlation analysis
- Visualization (optional)
"""

import os
import h5py
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# ============== CONFIGURATION ==============
# Đường dẫn tới các thư mục features
METHODS = {
    'method1_global': '/kaggle/working/features_v1',
    'method2_caption': '/kaggle/working/features_v2', 
    'method3_transe': '/kaggle/working/features_sota',
    'method4_full': '/kaggle/working/features_fullsota',
}

SPLITS = ['train', 'val', 'test']


def analyze_hdf5(path: str, split: str) -> Dict:
    """Phân tích một file HDF5"""
    filepath = os.path.join(path, f'MSVD_rel_{split}.hdf5')
    
    if not os.path.exists(filepath):
        return {
            'exists': False,
            'path': filepath
        }
    
    with h5py.File(filepath, 'r') as hf:
        video_ids = list(hf.keys())
        num_videos = len(video_ids)
        
        if num_videos == 0:
            return {
                'exists': True,
                'num_videos': 0
            }
        
        # Collect all embeddings
        all_embeddings = []
        shapes = []
        
        for vid in video_ids:
            data = hf[vid][:]
            all_embeddings.append(data)
            shapes.append(data.shape)
        
        all_embeddings = np.array(all_embeddings)
        
        # Get first frame embedding for uniqueness analysis
        first_frame_emb = all_embeddings[:, 0, :] if len(all_embeddings.shape) == 3 else all_embeddings
        
        # Count unique embeddings
        unique_embeddings = np.unique(first_frame_emb, axis=0)
        unique_ratio = len(unique_embeddings) / num_videos * 100
        
        # Statistics
        mean_val = np.mean(all_embeddings)
        std_val = np.std(all_embeddings)
        min_val = np.min(all_embeddings)
        max_val = np.max(all_embeddings)
        
        # Zero ratio
        zero_count = np.sum(np.all(first_frame_emb == 0, axis=1))
        zero_ratio = zero_count / num_videos * 100
        
        # Variance across videos
        video_variance = np.var(first_frame_emb, axis=0).mean()
        
        return {
            'exists': True,
            'num_videos': num_videos,
            'shape': shapes[0] if shapes else None,
            'unique_count': len(unique_embeddings),
            'unique_ratio': unique_ratio,
            'zero_count': zero_count,
            'zero_ratio': zero_ratio,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'video_variance': video_variance
        }


def print_separator():
    print("="*80)


def compare_all_methods():
    """So sánh tất cả các phương pháp"""
    print_separator()
    print("SO SÁNH TẤT CẢ PHƯƠNG PHÁP TẠO REL_FEATURE")
    print_separator()
    
    results = {}
    
    for method_name, method_path in METHODS.items():
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"Path: {method_path}")
        print("="*60)
        
        method_results = {}
        
        for split in SPLITS:
            stats = analyze_hdf5(method_path, split)
            method_results[split] = stats
            
            if not stats['exists']:
                print(f"\n  {split}: FILE NOT FOUND ({stats['path']})")
            elif stats['num_videos'] == 0:
                print(f"\n  {split}: EMPTY FILE")
            else:
                print(f"\n  {split}:")
                print(f"    Videos: {stats['num_videos']}")
                print(f"    Shape: {stats['shape']}")
                print(f"    Unique: {stats['unique_count']}/{stats['num_videos']} ({stats['unique_ratio']:.1f}%)")
                print(f"    Zero vectors: {stats['zero_count']} ({stats['zero_ratio']:.1f}%)")
                print(f"    Mean: {stats['mean']:.6f}")
                print(f"    Std: {stats['std']:.6f}")
                print(f"    Video variance: {stats['video_variance']:.6f}")
        
        results[method_name] = method_results
    
    # Summary comparison
    print("\n")
    print_separator()
    print("SUMMARY COMPARISON")
    print_separator()
    
    print("\n{:<20} | {:>12} | {:>12} | {:>12} | {:>12}".format(
        "Method", "Train Uniq%", "Val Uniq%", "Test Uniq%", "Avg Uniq%"
    ))
    print("-" * 75)
    
    for method_name in METHODS.keys():
        if method_name not in results:
            continue
        
        method_results = results[method_name]
        
        train_uniq = method_results['train'].get('unique_ratio', 0) if method_results['train']['exists'] else 0
        val_uniq = method_results['val'].get('unique_ratio', 0) if method_results['val']['exists'] else 0
        test_uniq = method_results['test'].get('unique_ratio', 0) if method_results['test']['exists'] else 0
        
        valid_ratios = [r for r in [train_uniq, val_uniq, test_uniq] if r > 0]
        avg_uniq = np.mean(valid_ratios) if valid_ratios else 0
        
        print("{:<20} | {:>11.1f}% | {:>11.1f}% | {:>11.1f}% | {:>11.1f}%".format(
            method_name, train_uniq, val_uniq, test_uniq, avg_uniq
        ))
    
    print("-" * 75)
    
    # Recommendations
    print("\n")
    print_separator()
    print("RECOMMENDATIONS")
    print_separator()
    
    print("""
    📊 Đánh giá các phương pháp:
    
    1. method1_global (V1 Original):
       ❌ Không nên dùng - Tất cả videos có cùng embedding
       ❌ Unique ratio ~0.1%
       ❌ Không học được thông tin per-video
    
    2. method2_caption (V2 Caption Matching):
       ✅ Cải thiện đáng kể so với V1
       ✅ Unique ratio ~60-80%
       ✅ Đơn giản, dễ implement
       ⚠️ Chỉ dựa vào caption text
    
    3. method3_transe (TransE + Weighted):
       ✅✅ SOTA embeddings học được relational structure
       ✅ Weighted aggregation (entity > relation)
       ✅ Domain-specific thay vì general GloVe
       ⚠️ Cần training TransE
    
    4. method4_full (Object Detection + Scene Graph):
       ✅✅✅ Full SOTA approach theo paper
       ✅ Kết hợp visual (objects) + text (captions)
       ✅ Scene graph cho relationships
       ⚠️ Cần pre-extracted features
       ⚠️ Phức tạp nhất
    
    🎯 Khuyến nghị:
    - Nếu cần nhanh: method2_caption
    - Nếu muốn SOTA: method3_transe hoặc method4_full
    - Tránh dùng: method1_global
    """)
    
    return results


def main():
    compare_all_methods()


if __name__ == '__main__':
    main()
