export interface GitHubProjectFile {
  name: string;
  path: string;
  size: number;
  downloadUrl?: string;
}

export interface GitHubProjectItem {
  id: string;
  name: string;
  fullName: string;
  description: string;
  language: string;
  stars: number;
  forks: number;
  url: string;
  category: 'Autonomous Agents' | 'RAG & NLP' | 'Full Stack & Auth' | 'Machine Learning' | 'Data Science & BI' | 'Cloud & Systems';
  highlightFiles: {
    filename: string;
    language: string;
    code: string;
  }[];
  filesList: string[];
}

export interface HermesTaskItem {
  id: string;
  title: string;
  status: 'In Progress' | 'Operational' | 'Optimized' | 'Staging';
  category: 'Hermes Core' | 'Gateway RPC' | 'Tools & Skills' | 'Prompt Caching';
  description: string;
  model: string;
  latency: string;
}

export interface GcpServiceItem {
  id: string;
  name: string;
  category: 'Serverless' | 'Analytics & AI' | 'Storage' | 'Distributed Compute' | 'Databases';
  region: string;
  status: 'Healthy' | 'Active' | 'Connected';
  description: string;
  metrics: string;
  tools: string[];
}

export const IBRAHIM_PROFILE = {
  name: 'Ibrahim Abdelsattar',
  nameAr: 'إبراهيم عبد الستار',
  title: 'Full Stack AI Engineer & Systems Architect',
  bio: 'Full Stack AI Engineer specializing in Deep Learning, NLP, Multimodal Systems, RAG architectures, and scalable cloud solutions.',
  avatarUrl: 'https://avatars.githubusercontent.com/u/152749334?v=4',
  githubUrl: 'https://github.com/IbrahimAbdelsattar',
  githubUsername: 'IbrahimAbdelsattar',
  stats: {
    publicRepos: 41,
    followers: 14,
    following: 17,
    activeProjects: 8,
    gcpServices: 6,
  },
  skills: [
    'PyTorch',
    'Transformers',
    'RAG & Vector DBs',
    'Whisper ASR',
    'Qwen LLM',
    'FastAPI',
    'TypeScript',
    'React',
    'Next.js',
    'Google Cloud (BigQuery, Cloud Run, GCS, Dataproc)',
    'Docker',
    'Tailwind CSS',
  ],
};

export const IBRAHIM_GITHUB_PROJECTS: GitHubProjectItem[] = [
  {
    id: 'hermes-agent',
    name: 'hermes-agent',
    fullName: 'IbrahimAbdelsattar/hermes-agent',
    description: 'Personal AI agent with terminal execution, voice sentinel, dynamic toolset resolution, and prompt-caching turn loop.',
    language: 'Python',
    stars: 12,
    forks: 3,
    url: 'https://github.com/IbrahimAbdelsattar/hermes-agent',
    category: 'Autonomous Agents',
    filesList: ['run_agent.py', 'model_tools.py', 'toolsets.py', 'cli.py', 'gateway/run.py', 'web/src/pages/JarvisCallPage.tsx', 'web/src/components/JarvisCoreWidget.tsx'],
    highlightFiles: [
      {
        filename: 'model_tools.py',
        language: 'python',
        code: `"""Hermes Agent Tool Orchestration & Built-in Function Dispatch"""
import inspect
from typing import Any, Dict, List, Optional
from tools.registry import get_registered_tools, ToolDefinition

def discover_builtin_tools(enabled_toolsets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Discover active model tools respecting session-level toolsets and TTL caching."""
    tools = []
    registered = get_registered_tools()
    for name, tool_def in registered.items():
        if enabled_toolsets and tool_def.toolset not in enabled_toolsets:
            continue
        tools.append(tool_def.to_openai_schema())
    return tools

async def handle_function_call(tool_name: str, arguments: Dict[str, Any], context: Any) -> Any:
    """Safely execute registered core tool with profile scope bindings."""
    tool = get_registered_tools().get(tool_name)
    if not tool:
        raise ValueError(f"Unknown tool: {tool_name}")
    return await tool.execute(arguments, context=context)`,
      },
      {
        filename: 'JarvisCoreWidget.tsx',
        language: 'typescript',
        code: `// J.A.R.V.I.S. Core Executive Assistant - Ibrahim Abdelsattar
export const JarvisCoreWidget: React.FC<JarvisCoreWidgetProps> = ({ biometrics, onSendMessage }) => {
  const [viewMode, setViewMode] = useState<'chat' | 'projects' | 'cloud' | 'graph' | 'split'>('chat');
  // Dynamic code extraction, Hermes task orchestration & GCP monitoring
  return (
    <div className="w-full h-full flex flex-col rounded-xl bg-[#040d1a]/95 border border-[#00f0ff]/30">
      <TelemetryHeaderBar profile={IBRAHIM_PROFILE} />
      <FocusAndPomodoroBar targetMinutes={60} />
      {viewMode === 'projects' ? <ProjectsExplorer repos={IBRAHIM_GITHUB_PROJECTS} /> : <ChatView />}
    </div>
  );
};`,
      },
    ],
  },
  {
    id: 'mr-nlp-rag',
    name: 'MR-NLP-Robust-RAG-Chatbot',
    fullName: 'IbrahimAbdelsattar/MR-NLP-Robust-RAG-Chatbot',
    description: 'Production-grade multimodal voice & document RAG with Whisper ASR, Qwen 1.5-1.8B in 4-bit, and multi-tier embedding failovers.',
    language: 'Python',
    stars: 24,
    forks: 7,
    url: 'https://github.com/IbrahimAbdelsattar/MR-NLP-Robust-RAG-Chatbot',
    category: 'RAG & NLP',
    filesList: ['app.py', 'config.py', 'rag_system.py', 'embedding_systems.py', 'model_manager.py', 'document_processor.py', 'setup.py', 'README.md'],
    highlightFiles: [
      {
        filename: 'rag_system.py',
        language: 'python',
        code: `"""Core RAG retrieval engine with hybrid dense/sparse fallback scoring"""
import logging
from typing import List, Dict, Any

class RobustRAGSystem:
    def __init__(self, model_manager, collection_name: str, embedding_method: str):
        self.model_manager = model_manager
        self.collection_name = collection_name
        self.embedding_method = embedding_method
        self.similarity_threshold = 0.70

    def generate_rag_response(self, query: str, use_rag: bool = True, top_k: int = 3) -> Dict[str, Any]:
        """Query vector database with failover embeddings, assemble context, and synthesize answer."""
        retrieved_docs = self.retriever.search(query, top_k=top_k, threshold=self.similarity_threshold)
        if not retrieved_docs:
            return {"answer": "No factual match in knowledge base.", "sources": []}
        
        context = "\\n\\n".join([d["content"] for d in retrieved_docs])
        prompt = f"Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:"
        answer = self.model_manager.generate_response(prompt)
        return {"answer": answer, "sources": retrieved_docs, "context_length": len(context)}`,
      },
      {
        filename: 'embedding_systems.py',
        language: 'python',
        code: `"""Multi-Tier Adaptive Fallback Embeddings (SentenceTransformers -> MiniLM -> TF-IDF)"""
class AdaptiveEmbeddingChain:
    def __init__(self):
        self.primary_model = "sentence-transformers/all-mpnet-base-v2"
        self.secondary_model = "sentence-transformers/all-MiniLM-L6-v2"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self._encode_primary(texts)
        except Exception as primary_err:
            logging.warning(f"Primary embedder failed: {primary_err}. Falling back to MiniLM.")
            try:
                return self._encode_secondary(texts)
            except Exception as secondary_err:
                logging.error(f"Fallback to sparse TF-IDF vectors: {secondary_err}")
                return self._encode_tfidf(texts)`,
      },
    ],
  },
  {
    id: 'dual-site-clerk-auth',
    name: 'dual-site-clerk-auth',
    fullName: 'IbrahimAbdelsattar/dual-site-clerk-auth',
    description: 'Enterprise multi-domain single-sign-on (SSO) architecture with Clerk, Next.js edge middleware, and shared cross-domain session routing.',
    language: 'TypeScript',
    stars: 18,
    forks: 3,
    url: 'https://github.com/IbrahimAbdelsattar/dual-site-clerk-auth',
    category: 'Full Stack & Auth',
    filesList: ['middleware.ts', 'next.config.js', 'package.json', 'src/app/page.tsx', 'src/lib/clerk-provider.tsx'],
    highlightFiles: [
      {
        filename: 'middleware.ts',
        language: 'typescript',
        code: `import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

const isPublicRoute = createRouteMatcher(['/', '/sign-in(.*)', '/sign-up(.*)', '/api/health']);

export default clerkMiddleware(async (auth, req) => {
  const { userId } = await auth();
  const hostname = req.headers.get('host') || '';

  // Multi-tenant dual site domain routing
  if (hostname.includes('portal.domain.com')) {
    if (!userId && !isPublicRoute(req)) {
      return auth().redirectToSignIn({ returnBackUrl: req.url });
    }
  }

  return NextResponse.next();
});

export const config = {
  matcher: ['/((?!.*\\\\..*|_next).*)', '/', '/(api|trpc)(.*)'],
};`,
      },
    ],
  },
  {
    id: 'arabic-sentiment',
    name: 'Arabic-Sentiment-Analysis',
    fullName: 'IbrahimAbdelsattar/Arabic-Sentiment-Analysis',
    description: 'State-of-the-art Arabic text and dialect sentiment classifier with Tashkeel removal, morphological normalization, and transformer embeddings.',
    language: 'Jupyter Notebook',
    stars: 15,
    forks: 4,
    url: 'https://github.com/IbrahimAbdelsattar/Arabic-Sentiment-Analysis',
    category: 'RAG & NLP',
    filesList: ['app.py', 'Requirements.txt', 'Arabic_Sentiment.ipynb', 'preprocessing.py', 'README.md'],
    highlightFiles: [
      {
        filename: 'preprocessing.py',
        language: 'python',
        code: `import re
import pyarabic.araby as araby

def clean_arabic_text(text: str) -> str:
    """Normalize Arabic characters, remove diacritics (Tashkeel), elongation (Tatweel)."""
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\\u0600-\\u06FF\\s]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()`,
      },
    ],
  },
  {
    id: 'credit-card-fraud',
    name: 'Credit-card-Fraud-Detection',
    fullName: 'IbrahimAbdelsattar/Credit-card-Fraud-Detection',
    description: 'High-frequency transaction fraud detection utilizing SMOTE imbalance handling, PCA feature representation, and XGBoost anomaly classification.',
    language: 'Jupyter Notebook',
    stars: 18,
    forks: 6,
    url: 'https://github.com/IbrahimAbdelsattar/Credit-card-Fraud-Detection',
    category: 'Machine Learning',
    filesList: ['app.py', 'requirements.txt', 'fraud_detection.ipynb', 'README.md'],
    highlightFiles: [
      {
        filename: 'app.py',
        language: 'python',
        code: `import streamlit as st
import numpy as np
import joblib

model = joblib.load('fraud_model_xgboost.pkl')

st.title("🛡️ Real-Time Credit Card Fraud Sentinel")
amount = st.number_input("Transaction Amount ($USD)", min_value=0.01, value=150.0)
v_features = [st.slider(f"PCA Latent Component V{i}", -5.0, 5.0, 0.0) for i in range(1, 29)]

if st.button("Score Transaction Anomaly"):
    vector = np.array([[amount] + v_features])
    prob = model.predict_proba(vector)[0][1]
    if prob > 0.65:
        st.error(f"🚨 High Risk of Fraud Detected! Anomaly Probability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Verified Authentic Transaction. Risk Score: {prob*100:.2f}%")`,
      },
    ],
  },
  {
    id: 'supplymind-ai',
    name: 'SupplyMindAI',
    fullName: 'IbrahimAbdelsattar/SupplyMindAI',
    description: 'Intelligent multi-tenant supply chain management platform with demand forecasting and inventory replenishment modeling.',
    language: 'Jupyter Notebook',
    stars: 14,
    forks: 2,
    url: 'https://github.com/IbrahimAbdelsattar/SupplyMindAI',
    category: 'Data Science & BI',
    filesList: ['AI_ARCHITECTURE.md', 'AGENTS.md', 'API.md', 'supply_chain_optimization.ipynb'],
    highlightFiles: [
      {
        filename: 'AI_ARCHITECTURE.md',
        language: 'markdown',
        code: `# SupplyMind AI Architecture
- **Tenant Isolation**: Row-Level Security (RLS) PostgreSQL + pgvector
- **Demand Forecasting**: Temporal Fusion Transformers & Prophet forecasting
- **Inventory Safety Stock**: Statistical safety margins with Lead Time distribution
- **Autonomous Agent**: Hermes Agent webhook integration for supply re-orders`,
      },
    ],
  },
  {
    id: 'mesdaq-ai',
    name: 'Mesdaq_AI',
    fullName: 'IbrahimAbdelsattar/Mesdaq_AI',
    description: 'Financial intelligence dashboard with quantitative signal detection and market data streaming pipelines.',
    language: 'TypeScript',
    stars: 11,
    forks: 2,
    url: 'https://github.com/IbrahimAbdelsattar/Mesdaq_AI',
    category: 'Full Stack & Auth',
    filesList: ['api_schemas.py', 'DEPLOYMENT.md', 'README.md', 'src/components/Chart.tsx'],
    highlightFiles: [
      {
        filename: 'api_schemas.py',
        language: 'python',
        code: `from pydantic import BaseModel
from typing import List, Optional

class StockSignal(BaseModel):
    ticker: str
    price: float
    rsi_14: float
    macd_delta: float
    sentiment_score: float
    recommendation: str`,
      },
    ],
  },
  {
    id: 'numerix',
    name: 'Numerix',
    fullName: 'IbrahimAbdelsattar/Numerix',
    description: 'Interactive mathematical computing suite for numerical analysis, matrix decomposition, and calculus visualization.',
    language: 'TypeScript',
    stars: 7,
    forks: 1,
    url: 'https://github.com/IbrahimAbdelsattar/Numerix',
    category: 'Full Stack & Auth',
    filesList: ['DEPLOY.md', 'README.md', 'components.json', 'src/lib/matrix.ts'],
    highlightFiles: [
      {
        filename: 'src/lib/matrix.ts',
        language: 'typescript',
        code: `export function gaussianElimination(matrix: number[][]): number[] {
  const n = matrix.length;
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(matrix[k][i]) > Math.abs(matrix[maxRow][i])) maxRow = k;
    }
    [matrix[i], matrix[maxRow]] = [matrix[maxRow], matrix[i]];
    for (let k = i + 1; k < n; k++) {
      const c = -matrix[k][i] / matrix[i][i];
      for (let j = i; j <= n; j++) {
        matrix[k][j] = i === j ? 0 : matrix[k][j] + c * matrix[i][j];
      }
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = matrix[i][n] / matrix[i][i];
    for (let k = i - 1; k >= 0; k--) {
      matrix[k][n] -= matrix[k][i] * x[i];
    }
  }
  return x;
}`,
      },
    ],
  },
];

export const IBRAHIM_HERMES_TASKS: HermesTaskItem[] = [
  {
    id: 'task-01',
    title: 'Prompt Caching Optimization & Prefix Byte Stability',
    status: 'Operational',
    category: 'Prompt Caching',
    description: 'Enforce static prefix token ordering across turns to maximize Anthropic/OpenAI prompt cache hits and minimize token costs.',
    model: 'Claude 3.7 Sonnet / GPT-4o',
    latency: '820ms',
  },
  {
    id: 'task-02',
    title: 'Live Voice Sentinel with Nabra TTS & Whisper ASR',
    status: 'In Progress',
    category: 'Gateway RPC',
    description: 'Real-time WebSocket streaming audio exchange with Egyptian Arabic dialect synthesis and turn arbitration.',
    model: 'Whisper + Nabra TTS',
    latency: '450ms',
  },
  {
    id: 'task-03',
    title: 'Autonomous Toolset Discovery & Session Filter',
    status: 'Optimized',
    category: 'Tools & Skills',
    description: 'Dynamic gating of terminal, filesystem, browser automation, and MCP tools based on session surface capabilities.',
    model: 'Hermes Core Engine',
    latency: '12ms',
  },
  {
    id: 'task-04',
    title: 'Multi-Surface Gateway Dispatch (CLI, Desktop, Web)',
    status: 'Operational',
    category: 'Hermes Core',
    description: 'Unified JSON-RPC gateway orchestrating session state across Terminal UI, Electron Desktop, and React Web Dashboard.',
    model: 'Hermes Gateway',
    latency: '5ms',
  },
];

export const IBRAHIM_GCP_SERVICES: GcpServiceItem[] = [
  {
    id: 'bigquery',
    name: 'Google BigQuery',
    category: 'Analytics & AI',
    region: 'us-central1 / europe-west1',
    status: 'Connected',
    description: 'Petabyte data warehousing, BigQuery ML vector search embeddings, BigFrames ML models, and real-time streaming SQL queries.',
    metrics: 'Active Queries • 100% Availability',
    tools: ['bigquery-sql', 'bigquery-bigframes', 'bigquery-ai-ml'],
  },
  {
    id: 'cloud-run',
    name: 'Google Cloud Run',
    category: 'Serverless',
    region: 'us-central1',
    status: 'Active',
    description: 'Fully-managed serverless containers hosting Hermes Gateway WebSocket daemon and MR-NLP RAG microservices.',
    metrics: 'Auto-Scaling 0→10 • 120ms Cold Start',
    tools: ['cloudrun-deploy', 'cloudrun-monitoring'],
  },
  {
    id: 'gcs',
    name: 'Google Cloud Storage (GCS)',
    category: 'Storage',
    region: 'Multi-Region (US/EU)',
    status: 'Healthy',
    description: 'Object storage buckets holding pre-trained model weights (Qwen, MARBERT), vector index snapshots, and RAG document repositories.',
    metrics: '99.999999999% Durability',
    tools: ['google-cloud-storage-basics', 'gcsfuse'],
  },
  {
    id: 'dataproc',
    name: 'Google Dataproc Serverless',
    category: 'Distributed Compute',
    region: 'us-central1',
    status: 'Active',
    description: 'Managed Apache Spark clusters executing distributed feature engineering, SMOTE dataset rebalancing, and large NLP corpus tokenization.',
    metrics: 'Auto-Tuned Spark 3.5 Executors',
    tools: ['gcp-spark', 'gcp-data-pipelines'],
  },
  {
    id: 'cloud-sql',
    name: 'Cloud SQL PostgreSQL',
    category: 'Databases',
    region: 'us-central1',
    status: 'Connected',
    description: 'Managed PostgreSQL with pgvector extension powering tenant RLS, hybrid semantic search, and Clerk SSO session stores.',
    metrics: '99.99% SLA • Automatic Backups',
    tools: ['datacloud_cloud-sql_remote'],
  },
  {
    id: 'vertex-ai',
    name: 'Vertex AI & Gemini API',
    category: 'Analytics & AI',
    region: 'global',
    status: 'Connected',
    description: 'Google Gemini 1.5 Pro / Flash foundation models for multimodal reasoning, code generation, and complex planning.',
    metrics: 'High Rate Limits • Function Calling Enabled',
    tools: ['vertex-gemini', 'grounded-search'],
  },
];
