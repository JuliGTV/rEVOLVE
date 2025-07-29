#!/usr/bin/env python3
"""
Simple performance test to demonstrate thread-based evaluation improvements
"""

import time
import asyncio
import concurrent.futures
import sys
import os

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from example_problems.heilbronn_triangles.evaluation import evaluate_heilbronn_triangles
from example_problems.heilbronn_triangles.spec import INITIAL_CODE

def test_sequential_evaluations(num_evaluations=5):
    """Test sequential evaluations (old behavior)"""
    print(f"Testing {num_evaluations} sequential evaluations...")
    start = time.time()
    
    results = []
    for i in range(num_evaluations):
        result = evaluate_heilbronn_triangles(INITIAL_CODE, 11)
        results.append(result.fitness)
        print(f"  Eval {i+1}: {result.fitness:.6f}")
    
    end = time.time()
    print(f"Sequential total time: {end - start:.2f} seconds")
    print(f"Average per evaluation: {(end - start) / num_evaluations:.2f} seconds")
    return end - start

async def test_async_evaluations(num_evaluations=5):
    """Test async evaluations with thread pool (new behavior)"""
    print(f"\nTesting {num_evaluations} async evaluations...")
    start = time.time()
    
    # Create thread pool
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def evaluate_async():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, evaluate_heilbronn_triangles, INITIAL_CODE, 11)
    
    # Run evaluations concurrently
    tasks = [evaluate_async() for _ in range(num_evaluations)]
    results = await asyncio.gather(*tasks)
    
    end = time.time()
    
    for i, result in enumerate(results):
        print(f"  Eval {i+1}: {result.fitness:.6f}")
    
    print(f"Async total time: {end - start:.2f} seconds")
    print(f"Average per evaluation: {(end - start) / num_evaluations:.2f} seconds")
    
    executor.shutdown(wait=True)
    return end - start

async def main():
    """Main performance comparison"""
    print("=== Heilbronn Triangles Evaluation Performance Test ===\n")
    
    # Test sequential
    sequential_time = test_sequential_evaluations(5)
    
    # Test async
    async_time = await test_async_evaluations(5)
    
    # Compare results
    print(f"\n=== Performance Comparison ===")
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Async time: {async_time:.2f} seconds")
    print(f"Speedup: {sequential_time / async_time:.2f}x")
    print(f"Efficiency: {(sequential_time / async_time) * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())