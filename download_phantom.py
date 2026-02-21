"""
PHANTOM Dataset Downloader and Loader

PHANTOM: A Benchmark for Hallucination Detection in Financial Long-Context QA
Paper: NeurIPS 2024 (November 2024)
Link: https://openreview.net/forum?id=5YQAo0S3Hm

This dataset is specifically designed for hallucination detection in 
financial document QA - perfectly matching our research domain.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Known PHANTOM dataset sources
PHANTOM_SOURCES = {
    # Primary: HuggingFace Hub (most likely location)
    "huggingface": "https://huggingface.co/datasets/phantom-benchmark/phantom",
    # Alternative: GitHub repository
    "github": "https://github.com/phantom-benchmark/phantom",
    # OpenReview supplementary
    "openreview": "https://openreview.net/forum?id=5YQAo0S3Hm"
}

DATA_DIR = Path("data/phantom")


def download_from_huggingface():
    """Download PHANTOM dataset from HuggingFace"""
    try:
        from datasets import load_dataset
        
        print("Attempting to load PHANTOM from HuggingFace...")
        
        # Try common naming conventions
        possible_names = [
            "phantom-benchmark/phantom",
            "PHANTOM/financial-hallucination",
            "phantom-financial-qa",
            "neurips2024/phantom"
        ]
        
        for name in possible_names:
            try:
                dataset = load_dataset(name)
                print(f"Successfully loaded: {name}")
                return dataset
            except Exception:
                continue
        
        print("Dataset not found with common names.")
        print("Please check https://openreview.net/forum?id=5YQAo0S3Hm for the official dataset link.")
        return None
        
    except ImportError:
        print("Install datasets library: pip install datasets")
        return None


def create_phantom_compatible_format(custom_dataset_path: str = "data/processed/hallucination_test_dataset.json"):
    """
    Convert our custom dataset to PHANTOM-compatible format.
    This allows benchmarking on both datasets with the same code.
    """
    
    print("Creating PHANTOM-compatible format from custom dataset...")
    
    with open(custom_dataset_path, 'r') as f:
        custom_data = json.load(f)
    
    phantom_format = {
        "metadata": {
            "name": "Financial RAG Hallucination Dataset (PHANTOM-compatible)",
            "description": "Custom dataset formatted to match PHANTOM benchmark structure",
            "source": "SEC 10-K filings (2024-2025)",
            "original_format": "custom",
            "phantom_compatible": True
        },
        "data": []
    }
    
    for example in custom_data['examples']:
        # Convert to PHANTOM-style format
        phantom_entry = {
            "id": f"custom_{example['id']}",
            "question": example['question'],
            "company": example['company'],
            "document_type": "10-K",
            
            # PHANTOM typically has these fields
            "context": None,  # Will be filled by RAG retrieval
            
            # Grounded response
            "grounded": {
                "response": example['grounded_response'],
                "label": "grounded",
                "is_hallucinated": False
            },
            
            # Hallucinated response
            "hallucinated": {
                "response": example['hallucinated_response'],
                "label": "hallucinated", 
                "is_hallucinated": True,
                "hallucination_type": example.get('hallucination_type', 'unknown')
            },
            
            # Metadata
            "difficulty": example.get('difficulty', 'medium'),
            "domain": example.get('domain', 'financial')
        }
        
        phantom_format["data"].append(phantom_entry)
    
    # Save PHANTOM-compatible format
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "phantom_compatible_dataset.json"
    
    with open(output_path, 'w') as f:
        json.dump(phantom_format, f, indent=2)
    
    print(f"Saved PHANTOM-compatible dataset to: {output_path}")
    print(f"Total examples: {len(phantom_format['data'])}")
    
    return phantom_format


def load_phantom_dataset(path: Optional[str] = None) -> Dict:
    """
    Load PHANTOM dataset from file or download.
    
    Returns dataset in standardized format for benchmarking.
    """
    
    # Check for local PHANTOM dataset
    local_paths = [
        DATA_DIR / "phantom_dataset.json",
        DATA_DIR / "phantom_compatible_dataset.json",
        Path("data/phantom/train.json"),
        Path("data/phantom/test.json")
    ]
    
    if path:
        local_paths.insert(0, Path(path))
    
    for p in local_paths:
        if p.exists():
            print(f"Loading PHANTOM dataset from: {p}")
            with open(p, 'r') as f:
                return json.load(f)
    
    print("PHANTOM dataset not found locally.")
    print("\nTo use PHANTOM dataset:")
    print("1. Visit: https://openreview.net/forum?id=5YQAo0S3Hm")
    print("2. Download the dataset from supplementary materials")
    print("3. Place in data/phantom/ directory")
    print("\nAlternatively, using PHANTOM-compatible format from custom dataset...")
    
    return create_phantom_compatible_format()


def get_phantom_examples(dataset: Dict, split: str = "test") -> List[Dict]:
    """
    Extract examples from PHANTOM dataset for evaluation.
    
    Returns list of (question, response, context, label) tuples.
    """
    
    examples = []
    
    data = dataset.get('data', dataset.get('examples', []))
    
    for entry in data:
        # Handle both PHANTOM and custom format
        if 'grounded' in entry and 'hallucinated' in entry:
            # Our PHANTOM-compatible format
            examples.append({
                "id": entry['id'],
                "question": entry['question'],
                "response": entry['grounded']['response'],
                "is_hallucinated": False,
                "company": entry.get('company'),
                "difficulty": entry.get('difficulty')
            })
            examples.append({
                "id": entry['id'],
                "question": entry['question'],
                "response": entry['hallucinated']['response'],
                "is_hallucinated": True,
                "hallucination_type": entry['hallucinated'].get('hallucination_type'),
                "company": entry.get('company'),
                "difficulty": entry.get('difficulty')
            })
        else:
            # Original PHANTOM format (adjust based on actual structure)
            examples.append({
                "id": entry.get('id'),
                "question": entry.get('question'),
                "response": entry.get('response') or entry.get('answer'),
                "context": entry.get('context') or entry.get('document'),
                "is_hallucinated": entry.get('is_hallucinated') or entry.get('label') == 'hallucinated',
                "company": entry.get('company'),
                "difficulty": entry.get('difficulty')
            })
    
    return examples


def print_phantom_stats(dataset: Dict):
    """Print statistics about the PHANTOM dataset"""
    
    print("\n" + "="*60)
    print("PHANTOM DATASET STATISTICS")
    print("="*60)
    
    metadata = dataset.get('metadata', {})
    print(f"\nDataset: {metadata.get('name', 'PHANTOM')}")
    print(f"Description: {metadata.get('description', 'Financial Hallucination Detection')}")
    
    data = dataset.get('data', dataset.get('examples', []))
    print(f"\nTotal entries: {len(data)}")
    
    # Count by hallucination type
    if data and 'hallucinated' in data[0]:
        hall_types = {}
        for entry in data:
            h_type = entry.get('hallucinated', {}).get('hallucination_type', 'unknown')
            hall_types[h_type] = hall_types.get(h_type, 0) + 1
        
        print("\nHallucination types:")
        for h_type, count in sorted(hall_types.items()):
            print(f"  - {h_type}: {count}")
    
    # Count by company
    companies = {}
    for entry in data:
        company = entry.get('company', 'unknown')
        companies[company] = companies.get(company, 0) + 1
    
    print("\nBy company:")
    for company, count in sorted(companies.items()):
        print(f"  - {company}: {count}")
    
    # Count by difficulty
    difficulties = {}
    for entry in data:
        diff = entry.get('difficulty', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print("\nBy difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  - {diff}: {count}")
    
    print("="*60)


if __name__ == "__main__":
    print("PHANTOM Dataset Loader")
    print("NeurIPS 2024 - Financial Hallucination Detection Benchmark")
    print("-" * 60)
    
    # Try to load or create PHANTOM-compatible dataset
    dataset = load_phantom_dataset()
    
    if dataset:
        print_phantom_stats(dataset)
        
        # Show sample
        examples = get_phantom_examples(dataset)
        print(f"\nTotal evaluation examples: {len(examples)}")
        
        if examples:
            print("\nSample entry:")
            sample = examples[0]
            print(f"  Question: {sample['question'][:80]}...")
            print(f"  Is Hallucinated: {sample['is_hallucinated']}")
