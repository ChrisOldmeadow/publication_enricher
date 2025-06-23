#!/usr/bin/env python3
"""
Test script to demonstrate dashboard logging improvements
"""
import subprocess
import time
import os

def test_dashboard_logging():
    """Test the dashboard with a small sample to show logging"""
    
    # Create a small test file
    test_data = """Publication_ID,Output_Title,Ref_DOI
1,Test Publication 1,10.1000/test123
2,Test Publication 2,10.1000/test456
3,Anxiety and stress,10.1007/978-3-031-53976-3_10
4,Heart disease research,10.1016/j.jacc.2021.01.001
5,COVID-19 vaccines,10.1056/NEJMoa2034577"""
    
    test_file = 'test_logging_demo.csv'
    with open(test_file, 'w') as f:
        f.write(test_data)
    
    print("Created test file with 5 publications")
    print("Starting dashboard in background...")
    
    # Start dashboard in background
    dashboard_process = subprocess.Popen(['python', 'run_dashboard.py'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
    
    # Wait for dashboard to start
    time.sleep(3)
    print("Dashboard should be running at http://localhost:5001")
    
    # Start a test enrichment
    print("Starting test enrichment...")
    enrichment_cmd = [
        'python', 'multi_process_enricher.py', 
        test_file,
        '--processes', '1',
        '--batch-size', '2',
        '--max-concurrent', '2'
    ]
    
    try:
        # Run enrichment for a short time to generate logs
        enrichment_process = subprocess.Popen(enrichment_cmd, 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.STDOUT,
                                            text=True)
        
        print("Enrichment started. Check the dashboard at http://localhost:5001")
        print("You should now see:")
        print("1. Real-time progress bars")
        print("2. Processing rate (pub/s)")
        print("3. ETA estimates")
        print("4. Live enriched/failed counts")
        print("5. Recent log output")
        
        # Let it run for 30 seconds
        print("\nLetting it run for 30 seconds to demonstrate progress tracking...")
        time.sleep(30)
        
        # Stop the enrichment
        enrichment_process.terminate()
        print("Stopped enrichment process")
        
    except Exception as e:
        print(f"Error running enrichment: {e}")
    
    finally:
        # Clean up
        try:
            dashboard_process.terminate()
            print("Stopped dashboard")
        except:
            pass
        
        try:
            os.remove(test_file)
            # Remove any generated test files
            for f in ['test_logging_demo_enriched_mp.csv', 
                      'test_logging_demo_enriched_mp_failed.csv',
                      'test_logging_demo_enriched_mp_fuzzy.csv']:
                if os.path.exists(f):
                    os.remove(f)
        except:
            pass
        
        print("Cleaned up test files")

if __name__ == '__main__':
    print("=== Dashboard Logging Test ===")
    print("This will start the dashboard and run a small enrichment")
    print("to demonstrate the improved progress tracking.\n")
    
    input("Press Enter to start the test...")
    test_dashboard_logging()