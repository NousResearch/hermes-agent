import os
import requests
import base64

def fetch_github_repo_structure(repo_url: str):
    """Fetches the file tree of a GitHub repository to understand its architecture."""
    # URL format: https://github.com/owner/repo
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        tree = response.json().get("tree", [])
        files = [item['path'] for item in tree if item['type'] == 'blob'][:30] # Limit for context
        return f"Files found in {repo}:\n" + "\n".join(files)
    return "Error: Could not access the repository. Check the URL or branch name."

def read_github_file_content(repo_url: str, file_path: str):
    """Reads the content of a specific file from a GitHub repository."""
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        content_b64 = response.json().get("content", "")
        return base64.b64decode(content_b64).decode('utf-8')[:3000] # Context limit
    return "Error: File not found or inaccessible."
