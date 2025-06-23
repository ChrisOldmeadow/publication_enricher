#!/usr/bin/env python3
"""
Diagnostic script to check multiprocessing functionality
"""
import os
import sys
import json
import time
import subprocess
import multiprocessing as mp
from pathlib import Path

def test_basic_multiprocessing():
    """Test if basic multiprocessing works"""
    print("Testing basic multiprocessing...")
    
    def worker_func(x):
        return x * x
    
    try:
        with mp.Pool(processes=2) as pool:
            results = pool.map(worker_func, [1, 2, 3, 4])
        print(f"✓ Basic multiprocessing works: {results}")
        return True
    except Exception as e:
        print(f"✗ Basic multiprocessing failed: {e}")
        return False

def check_process_count():
    """Check system process limits"""
    try:
        max_processes = os.sysconf('SC_CHILD_MAX')
        print(f"System max processes: {max_processes}")
    except:
        print("Could not determine system process limits")

def test_subprocess_creation():
    """Test if we can create subprocesses"""
    print("\nTesting subprocess creation...")
    try:
        result = subprocess.run(['echo', 'test'], capture_output=True, text=True, timeout=5)
        print(f"✓ Subprocess works: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"✗ Subprocess failed: {e}")
        return False

def check_checkpoint_integrity():
    """Check if checkpoint files are valid"""
    print("\nChecking checkpoint files...")
    checkpoint_files = sorted(Path('.').glob('checkpoint_chunk_*.json'))
    
    total_processed = 0
    for i, checkpoint_file in enumerate(checkpoint_files):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                count = len(data.get('processed', []))
                total_processed += count
                print(f"✓ Checkpoint {i}: {count} publications processed")
        except Exception as e:
            print(f"✗ Checkpoint {i} ({checkpoint_file}) is corrupted: {e}")
    
    print(f"Total processed from checkpoints: {total_processed}")
    return total_processed

def test_enricher_import():
    """Test if we can import the enricher modules"""
    print("\nTesting module imports...")
    try:
        # Test if we can import the main modules
        sys.path.insert(0, '.')
        from publication_enricher.processor import PublicationProcessor
        from publication_enricher.api_client import APIClient
        print("✓ Core modules import successfully")
        return True
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        return False

def test_small_enrichment():
    """Test enrichment on a very small sample"""
    print("\nTesting small enrichment...")
    
    # Create a tiny test file
    test_data = """Publication_ID,Output_Title,Ref_DOI
1,Test Publication,10.1000/test123"""
    
    test_file = 'test_tiny.csv'
    with open(test_file, 'w') as f:
        f.write(test_data)
    
    try:
        # Run single process enrichment
        cmd = ['python', 'multi_process_enricher.py', test_file, '--processes', '1', '--batch-size', '1']
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Small enrichment completed successfully")
            print(f"Output: {result.stdout[-200:]}")  # Last 200 chars
        else:
            print("✗ Small enrichment failed")
            print(f"Error: {result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ Small enrichment timed out")
        return False
    except Exception as e:
        print(f"✗ Small enrichment error: {e}")
        return False
    finally:
        # Clean up
        try:
            os.remove(test_file)
            for f in Path('.').glob('test_tiny*'):
                f.unlink()
        except:
            pass

def check_system_resources():
    """Check available system resources"""
    print("\nChecking system resources...")
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU cores: {cpu_count}, Current usage: {cpu_percent}%")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.percent}% used, {memory.available // (1024**3)} GB available")
        
        # Check if we're running in a container or restricted environment
        if cpu_count <= 1:
            print("⚠ Warning: Only 1 CPU core detected - multiprocessing may be limited")
            
        if memory.available < 1024**3:  # Less than 1GB
            print("⚠ Warning: Low memory available")
            
    except ImportError:
        print("psutil not available - cannot check system resources")

def main():
    print("=== Publication Enricher Multiprocessing Diagnostic ===\n")
    
    # Run all diagnostic tests
    tests = [
        ("System Resources", check_system_resources),
        ("Process Count Limits", check_process_count),
        ("Basic Multiprocessing", test_basic_multiprocessing),
        ("Subprocess Creation", test_subprocess_creation),
        ("Module Imports", test_enricher_import),
        ("Checkpoint Integrity", check_checkpoint_integrity),
        ("Small Enrichment Test", test_small_enrichment)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print('='*50)
    
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "- INFO"
        print(f"{status:<8} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed < total:
        print("\n⚠ Issues detected. Check the failed tests above.")
    else:
        print("\n✓ All tests passed. Multiprocessing should work correctly.")

if __name__ == '__main__':
    main()