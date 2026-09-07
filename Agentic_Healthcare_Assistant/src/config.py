"""
Configuration module for the Agentic Healthcare Assistant.
Strictly configured for OpenRouter as the primary LLM provider.
Manages file paths, OpenRouter credentials, model configurations, and system defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "DataSet"
SRC_DIR = BASE_DIR / "src"
DATA_STORAGE_DIR = BASE_DIR / "data_storage"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
DATA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
EXCEL_RECORDS_PATH = DATASET_DIR / "records.xlsx"
RUNTIME_EXCEL_PATH = DATA_STORAGE_DIR / "runtime_records.xlsx"
APPOINTMENTS_DB_PATH = DATA_STORAGE_DIR / "appointments.json"
DOCTORS_DB_PATH = DATA_STORAGE_DIR / "doctors.json"
EVALUATION_LOG_PATH = LOGS_DIR / "evaluation_metrics.json"
AUDIT_LOG_PATH = LOGS_DIR / "agent_audit_traces.jsonl"

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# OpenRouter Credentials
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default OpenRouter Models
# Includes both standard and popular free-tier models on OpenRouter
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

POPULAR_OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o-mini"
]

# Embeddings dimension for local FAISS
EMBEDDING_DIM = 128

# Standard medical specialties
MEDICAL_SPECIALTIES = [
    "Nephrology",
    "Cardiology",
    "Endocrinology",
    "Pulmonology",
    "Family Medicine",
    "Geriatrics",
    "Neurology",
    "General Medicine"
]
