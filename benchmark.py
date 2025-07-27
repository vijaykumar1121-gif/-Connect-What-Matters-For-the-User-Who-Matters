#!/usr/bin/env python3
"""
Performance Benchmarking Script for Persona-Driven Document Intelligence System.
Tests model size, processing time, and constraint compliance.
"""

import os
import time
import json
import psutil
import torch
from pathlib import Path
from src.main import run_analysis
from config import TEST_CASES, CONSTRAINTS, MODEL_PATHS

def get_model_size_mb():
    """Calculate total model size in MB."""
    total_size = 0
    
    # Check for model files
    for model_path in MODEL_PATHS.values():
        if os.path.exists(model_path):
            total_size += os.path.getsize(model_path) / (1024 * 1024)
    
    # Estimate transformer model sizes
    transformer_sizes = {
        "sentence-transformers/all-MiniLM-L6-v2": 90,
        "t5-small": 242,
        "distilbert-base-cased-distilled-squad": 260
    }
    
    for model_name, size in transformer_sizes.items():
        total_size += size
    
    return total_size

def benchmark_processing_time(test_case_name, test_case):
    """Benchmark processing time for a test case."""
    print(f"\n🧪 Benchmarking: {test_case_name}")
    
    # Get test documents
    test_docs_dir = Path("test_docs")
    pdf_files = list(test_docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in test_docs/")
        return None
    
    # Measure processing time
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        output = run_analysis(
            [str(f) for f in pdf_files],
            test_case["persona"],
            test_case["job"]
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return {
            "test_case": test_case_name,
            "processing_time_seconds": processing_time,
            "memory_usage_mb": memory_usage,
            "num_documents": len(pdf_files),
            "success": True
        }
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "test_case": test_case_name,
            "processing_time_seconds": processing_time,
            "error": str(e),
            "success": False
        }

def check_constraints():
    """Check if the system meets all constraints."""
    print("🔍 Checking System Constraints...")
    
    # Check model size
    model_size_mb = get_model_size_mb()
    size_compliant = model_size_mb <= CONSTRAINTS["max_model_size_mb"]
    
    print(f"📦 Model Size: {model_size_mb:.1f}MB / {CONSTRAINTS['max_model_size_mb']}MB")
    print(f"   Status: {'✅ Compliant' if size_compliant else '❌ Exceeds Limit'}")
    
    # Check CPU-only requirement
    cpu_only = not torch.cuda.is_available()
    print(f"🖥️  CPU-Only: {'✅ Yes' if cpu_only else '❌ GPU Available'}")
    
    return {
        "model_size_mb": model_size_mb,
        "size_compliant": size_compliant,
        "cpu_only": cpu_only
    }

def run_benchmarks():
    """Run all benchmarks."""
    print("🚀 Starting Performance Benchmarks")
    print("=" * 50)
    
    # Check constraints
    constraints = check_constraints()
    
    # Run processing time benchmarks
    results = []
    for test_case_name, test_case in TEST_CASES.items():
        result = benchmark_processing_time(test_case_name, test_case)
        if result:
            results.append(result)
    
    # Analyze results
    print("\n📊 Benchmark Results")
    print("=" * 50)
    
    successful_runs = [r for r in results if r["success"]]
    
    if successful_runs:
        avg_time = sum(r["processing_time_seconds"] for r in successful_runs) / len(successful_runs)
        max_time = max(r["processing_time_seconds"] for r in successful_runs)
        avg_memory = sum(r["memory_usage_mb"] for r in successful_runs) / len(successful_runs)
        
        print(f"⏱️  Average Processing Time: {avg_time:.2f}s")
        print(f"⏱️  Maximum Processing Time: {max_time:.2f}s")
        print(f"💾 Average Memory Usage: {avg_memory:.1f}MB")
        
        # Check time constraint
        time_compliant = max_time <= CONSTRAINTS["max_processing_time_seconds"]
        print(f"⏰ Time Constraint: {'✅ Compliant' if time_compliant else '❌ Exceeds 60s'}")
    
    # Summary
    print("\n🎯 Constraint Compliance Summary")
    print("=" * 50)
    print(f"📦 Model Size: {'✅' if constraints['size_compliant'] else '❌'}")
    print(f"⏰ Processing Time: {'✅' if 'time_compliant' in locals() and time_compliant else '❌'}")
    print(f"🖥️  CPU-Only: {'✅' if constraints['cpu_only'] else '❌'}")
    
    # Save results
    benchmark_results = {
        "constraints": constraints,
        "benchmarks": results,
        "summary": {
            "total_tests": len(results),
            "successful_tests": len(successful_runs),
            "all_constraints_met": constraints['size_compliant'] and 
                                 ('time_compliant' in locals() and time_compliant) and 
                                 constraints['cpu_only']
        }
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\n💾 Results saved to: benchmark_results.json")
    
    return benchmark_results

if __name__ == "__main__":
    run_benchmarks() 