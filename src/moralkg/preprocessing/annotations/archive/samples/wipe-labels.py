import rootutils
import os

root = rootutils.setup_root(__file__, dotenv=True)

def wipe_labels() -> bool:
    """
    Wipe all data files from a HuggingFace dataset.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi
        dataset_name = "RIET-lab/moral-kg-sample-labels"    

        print(f"Attempting to delete all data files from HuggingFace dataset: {dataset_name}")
        
        # Initialize HF API
        api = HfApi()
        
        # List all files in the repository
        repo_files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset", token=os.getenv("HF_TOKEN"))
        
        # Filter files in the data/ directory
        data_files = [f for f in repo_files if f.startswith("data/")]
        
        if not data_files:
            print("No data files found to delete")
            return True
            
        print(f"Found {len(data_files)} data files to delete")
        
        # Delete each data file
        for file_path in data_files:
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=os.getenv("HF_TOKEN"),
                    commit_message=f"Delete {file_path}"
                )
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        
        print("HuggingFace label dataset data wipe")
        return True
        
    except ImportError:
        print(f"Error: HuggingFace label dataset data wipe: {Exception('huggingface_hub not installed')}")
        return False
    except Exception as e:
        print("Error: HuggingFace label dataset data wipe", e)
        return False


if __name__ == "__main__":
    exit(wipe_labels())