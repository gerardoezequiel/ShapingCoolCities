from huggingface_hub import HfApi, hf_hub_download
import os
from tqdm.notebook import tqdm

def download_from_huggingface(repo_id, repo_type, folder_path=None, file_paths=None, local_dir="."):
    """
    Download an entire folder or specific files from a Hugging Face dataset repository.

    Parameters:
    repo_id (str): The ID of the repository (e.g., 'username/repo_name').
    repo_type (str): Type of the repo, either 'dataset' or 'model'.
    folder_path (str, optional): The path to the folder within the repository.
    file_paths (list, optional): List of file paths to download from the repository.
    local_dir (str): Local folder to download the data.
    """
    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    api = HfApi()

    # Determine the list of files to download
    if folder_path:
        # List all files in the repo and filter the ones within the folder path
        all_files = api.list_repo_files(repo_id, repo_type=repo_type)
        files_list = [f for f in all_files if f.startswith(folder_path)]
    elif file_paths:
        files_list = file_paths
    else:
        raise ValueError("Either folder_path or file_paths must be provided.")

    # Function to check if a file exists in the local directory tree
    def file_exists_in_local_tree(file_name, root_dir):
        for root, _, files in os.walk(root_dir):
            if file_name in files:
                return True
        return False

    # Download each specified file directly into the local_dir with a progress bar
    for file_path in tqdm(files_list, desc="Downloading files", unit="file"):
        # Extract the file name from the full path
        file_name = os.path.basename(file_path)

        # Adjust the local file path to save directly in local_dir
        local_file_path = os.path.join(local_dir, file_name)

        # Check if the file already exists in the local directory tree
        if file_exists_in_local_tree(file_name, local_dir):
            print(f"File '{file_name}' already exists in the directory. Skipping download.")
        else:
            print(f"Downloading file: {file_name}")
            try:
                hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=file_path, local_dir=local_dir)
            except Exception as e:
                print(f"Failed to download {file_name}. Error: {e}")