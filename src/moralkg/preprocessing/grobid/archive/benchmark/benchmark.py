import os
import time
import json
from datetime import datetime, timedelta
from grobid_client.grobid_client import GrobidClient

def benchmark_process(client: GrobidClient, process_name, input_path, output_path, n=64):
    """Run a specific Grobid process for PDF files and measure execution time"""
    start_time = time.time()
    client.process(
        process_name, 
        input_path, 
        output=output_path,
        n=n
    )
    end_time = time.time()
    return end_time - start_time

def format_time(seconds):
    """Format time in dd:hh:mm format"""
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"

def main():    
    config_path = "/opt/extra/avijit/projects/moralkg/data/scripts/grobid/config.json"
    input_path = "/opt/extra/avijit/projects/moralkg/data/scripts/grobid/benchmark/pdfs"
    output_path = "/opt/extra/avijit/projects/moralkg/data/scripts/grobid/benchmark/output"
    benchmark_dir = "/opt/extra/avijit/projects/moralkg/data/scripts/grobid/benchmark"
    pdf_count = 10
    
    # Make sure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize client
    client = GrobidClient(config_path=config_path)
    
    # Collect results
    results = {
        "timestamp": datetime.now().isoformat(),
        "pdf_count": pdf_count,
        "processes": {}
    }
    
    process = "processReferences"

    process_output = os.path.join(output_path, process)
    os.makedirs(process_output, exist_ok=True)
    
    execution_time = benchmark_process(client, process, input_path, process_output)
    avg_time_per_pdf = execution_time / pdf_count
    estimated_time_100k = avg_time_per_pdf * 100000
    
    results["processes"][process] = {
        "execution_time_seconds": execution_time,
        "avg_time_per_pdf_seconds": avg_time_per_pdf,
        "estimated_time_100k_pdfs_seconds": estimated_time_100k,
        "estimated_time_100k_pdfs_formatted": format_time(estimated_time_100k)
    }
    
    print(f"  Time for {pdf_count} PDFs: {execution_time:.2f} seconds")
    print(f"  Avg time per PDF: {avg_time_per_pdf:.2f} seconds")
    print(f"  Estimated time for 100K PDFs: {format_time(estimated_time_100k)}")
    
    # Write results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(benchmark_dir, f"benchmark_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
if __name__ == "__main__":
    main() 