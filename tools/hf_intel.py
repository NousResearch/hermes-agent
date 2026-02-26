import requests

def get_hf_model_info(model_id: str):
    """
    Fetches metadata, download counts, and technical tags for any model on Hugging Face.
    This helps Hermes act as an AI Research Assistant.
    """
    api_url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        data = response.json()
        model_name = data.get("id")
        downloads = data.get("downloads", 0)
        likes = data.get("likes", 0)
        tags = data.get("tags", [])[:5]
        pipeline = data.get("pipeline_tag", "unknown")
        
        return (f"Model Analysis: {model_name}\n"
                f"Downloads: {downloads}\n"
                f"Likes: {likes}\n"
                f"Task: {pipeline}\n"
                f"Tags: {', '.join(tags)}")
    
    return "Error: Model not found on Hugging Face Hub."
