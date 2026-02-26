import re
import io
import requests
import fitz  # PyMuPDF
from enum import Enum
from pydantic import BaseModel, Field

# 1. LLM'in anlayacağı çıktı formatlarını tanımlıyoruz
class OutputFormat(str, Enum):
    abstract = "abstract"
    eli5 = "eli5"
    methodology = "methodology"
    key_findings = "key_findings"
    comprehensive = "comprehensive"

# 2. Tool'un kabul edeceği girdilerin (parametrelerin) şeması
class ArxivResearchInput(BaseModel):
    url: str = Field(
        ..., 
        description="The ArXiv URL or raw ArXiv ID of the paper (e.g., 'https://arxiv.org/abs/2310.06825' or '2310.06825')."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.comprehensive,
        description="The desired format of the paper analysis. Choose 'abstract', 'eli5', 'methodology', 'key_findings', or 'comprehensive'."
    )

# 3. Aracın ana çalışma fonksiyonu
def arxiv_research(url: str, output_format: str = "comprehensive") -> str:
    """Fetches an academic paper from ArXiv, extracts the text, and prepares it for LLM analysis."""
    try:
        # URL'den ID'yi bul
        match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', url)
        if not match:
            return f"Error: Could not extract a valid ArXiv ID from {url}"
        arxiv_id = match.group(1)

        # PDF'i indir (Hafıza içi / In-memory)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()
        
        # PDF'i oku
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + "\n"
            
        # Token limitini korumak için metni temizle ve kırp
        text = re.sub(r'\s+', ' ', full_text)
        max_chars = 40000 # Yaklaşık 10k token limiti
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n...[TEXT TRUNCATED DUE TO LENGTH BUDGET]..."
            
        # Ajanın sistemine metni ve ne yapması gerektiğini döndür
        return f"""
Successfully retrieved ArXiv paper {arxiv_id}.
Please analyze the text below and provide the output strictly in the requested '{output_format}' format.

PAPER TEXT:
{text.strip()}
"""
        
    except requests.exceptions.RequestException as e:
        return f"Network Error: Failed to download PDF from ArXiv. It might be unavailable: {str(e)}"
    except Exception as e:
        return f"Parsing Error: Failed to parse PDF content: {str(e)}"

        # --- TOOL REGISTRY (SİSTEME KAYIT) ---
from tools.registry import registry

# Sistemin tam olarak beklediği OpenAI fonksiyon şeması
arxiv_schema = {
    "name": "arxiv_research",
    "description": "Fetches an academic paper from ArXiv, extracts the text, and prepares it for LLM analysis.",
    "parameters": ArxivResearchInput.model_json_schema()
}

registry.register(
    name="arxiv_research",
    toolset="web",
    schema=arxiv_schema,
    handler=arxiv_research
)