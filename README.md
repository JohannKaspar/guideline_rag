# Guideline RAG System

A Retrieval-Augmented Generation (RAG) system for medical guidelines that processes PDF documents, creates embeddings, and provides conversational access to the content.

## Project Structure

```
guideline_rag/
├── src/                          # Main source code package
│   ├── main.py                  # Main entry point module
│   ├── utils/                   # Shared utilities
│   │   ├── config.py            # Configuration management
│   │   ├── metadata.py          # Metadata processing utilities
│   │   ├── embeddings.py        # Centralized embedding model handling
│   │   └── clients.py           # GenAI Hub client setup
│   ├── core/                    # Core functionality modules
│   │   ├── vector_store.py      # Vector database operations
│   │   ├── document_processor.py # Document conversion and annotation
│   │   └── retrieval.py         # Retrieval and chat functionality
│   └── cli/                     # CLI command implementations
│       ├── embed.py             # Embedding command
│       ├── chat.py              # Chat command
│       └── process.py           # Document processing command
├── notebooks/                   # Jupyter notebooks for experimentation
├── data/                        # Data directories
│   ├── american_guidelines/     # American medical guidelines
│   ├── annotated/              # Processed and annotated documents
│   ├── converted/              # Converted documents
│   ├── pdfs/                   # Source PDF files
│   └── chroma_db/              # Vector database storage
├── pyproject.toml              # Package configuration and dependencies
├── backup/                     # Backup of old scripts
│   ├── embed_old.py            # Original embedding script
│   └── retrieval_chat_old.py   # Original chat interface
└── legacy/                     # Legacy files (for reference)
    ├── embed.py                # Original embedding script
    ├── process_doc.py          # Original document processing
    └── retrieval_chat.py       # Original chat interface
```

## Features

- **Document Processing**: Convert PDFs to structured format with table and image annotation
- **Embedding Generation**: Create vector embeddings using HuggingFace models
- **Vector Storage**: Store and retrieve embeddings using ChromaDB
- **Conversational Interface**: Chat with documents using retrieval-augmented generation
- **Modular Architecture**: Clean separation of concerns with reusable components

## Usage

### Installation

First, install the package in development mode:

```bash
uv pip install -e .
```

## CLI Commands

The system provides three main commands that work together in a typical workflow: **process** → **embed** → **chat**.

### Overview

1. **`process`** - Convert PDF documents to structured format using Docling
2. **`embed`** - Generate vector embeddings and store in database  
3. **`chat`** - Interactive conversational interface with the processed documents

### Command Line Interface

```bash
# Using the installed command (recommended)
guideline-rag <command>

# Get help
guideline-rag --help
```

### 1. Document Processing (`process`)

Converts PDF medical guidelines into structured JSON format using Docling's advanced document processing capabilities.

```bash
guideline-rag process
```

**What it does:**
- **PDF Conversion**: Uses Docling to extract text, tables, and images from PDFs
- **Image Processing**: Filters small images and generates AI descriptions using GPT-4.1-mini
- **Table Correction**: Corrects table HTML using vision models for better accuracy
- **Batch Processing**: Processes multiple documents in parallel with progress tracking
- **Error Handling**: Robust error handling with detailed logging and status tracking

**Input:** PDF files from `pdfs/` directory
**Output:** 
- Converted JSON files in `data/converted/`
- Annotated JSON files in `data/annotated/` (with AI image/table descriptions)
- Processing log in `convert.log`
- Status tracking in `doc_status.csv`

**Features:**
- Multiprocessing support (3 parallel processes by default)
- Two modes: convert-only or full annotation
- Progress tracking with visual progress bars
- Automatic retry and error recovery

### 2. Embedding Generation (`embed`)

Creates vector embeddings from processed documents and stores them in ChromaDB for efficient retrieval.

```bash
guideline-rag embed
```

**What it does:**
- **Document Chunking**: Splits annotated documents into optimal chunks for embedding
- **Vector Generation**: Creates embeddings using HuggingFace models (jinaai/jina-embeddings-v3)
- **Database Storage**: Stores embeddings in ChromaDB vector database
- **Deduplication**: Skips documents that are already embedded

**Input:** Annotated JSON files from `data/annotated/`
**Output:** Vector embeddings stored in `chroma_db/` directory

**Features:**
- Intelligent chunking with configurable token limits
- Metadata preservation for enhanced retrieval
- Automatic duplicate detection
- Progress tracking for large document sets

### 3. Interactive Chat (`chat`)

Provides a conversational interface to query the processed medical guidelines using retrieval-augmented generation (RAG).

```bash
guideline-rag chat
```

**What it does:**
- **Semantic Search**: Finds relevant document chunks using vector similarity
- **Context Assembly**: Combines retrieved chunks into coherent context
- **AI Response**: Generates answers using advanced language models
- **Interactive Interface**: Provides a user-friendly chat experience

**Features:**
- Real-time document retrieval
- Context-aware responses
- Source attribution for transparency
- Continuous conversation with memory

## Typical Workflow

### Complete Pipeline
```bash
# 1. Process PDF documents (one-time setup)
guideline-rag process

# 2. Generate embeddings (run after processing new documents)
guideline-rag embed

# 3. Start interactive chat
guideline-rag chat
```

### Example Session
```bash
# Process medical guidelines
$ guideline-rag process
Found 150 files to process.
Processing documents: 100%|████████| 150/150 [2:30:45<00:00, 60.30s/it]
Processing complete: 147/150 files successful

# Create embeddings
$ guideline-rag embed
Processing document_1.json...
Added 45 chunks for document_1.pdf
Processing document_2.json...
Added 32 chunks for document_2.pdf
...

# Start chat interface
$ guideline-rag chat
🏥 Medical Guidelines Chat Assistant
Type 'quit' to exit, 'help' for commands.

You: What are the recommendations for treating hypertension?
Assistant: Based on the medical guidelines, here are the key recommendations...
```

## Configuration

The system can be configured via environment variables:

### Environment Variables

- **`GUIDELINE_RAG_BASE_PATH`**: Custom base path for data directories
  ```bash
  export GUIDELINE_RAG_BASE_PATH="work/guideline_rag"
  ```
  If set, the system will use `{base_path}/converted/`, `{base_path}/annotated/`, and `{base_path}/pdfs/` instead of the default `data/` directories.

- **`CONVERT_ONLY_MODE`**: Enable convert-only mode (skips AI annotation)
  ```bash
  export CONVERT_ONLY_MODE="true"
  ```
  When enabled, the system only converts PDFs to JSON without AI-powered image descriptions or table corrections.

- **Model settings**: Additional settings configurable in `src/utils/config.py`

### AI Model Backend

**Current Implementation**: The system currently uses **SAP GenAI Hub** for AI model access, including:
- GPT-4.1-mini for image annotation and table correction
- Various language models for chat functionality
- Centralized model management through SAP's enterprise AI platform

**Contributing**: We welcome contributions to add support for **LiteLLM**, which would enable:
- Support for multiple AI providers (OpenAI, Anthropic, Azure, etc.)
- More flexible model selection and configuration
- Easier deployment in different environments
- Cost optimization through provider switching

If you're interested in contributing LiteLLM support, please see the contributing guidelines or open an issue to discuss the implementation approach.

## File Structure After Processing

```
data/
├── converted/          # Intermediate converted documents (JSON)
├── annotated/          # Fully processed documents with AI annotations
└── american_guidelines/ # Additional guideline data

chroma_db/              # Vector database storage
├── chroma.sqlite3      # Database file
└── ...                 # Index files

logs/                   # Processing logs
convert.log            # Document processing log
doc_status.csv         # Processing status tracking
```
