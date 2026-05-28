# Change Log

## Version 1.0 - Qdrant Migration
Author: Siddhidatri
Date: 2026-05-26

### Added
- Docker based Qdrant setup
- Local vector database storage
- Metadata tagging system
- Query classification layer
- Qdrant ingestion pipeline
- Qdrant retrieval pipeline

### Modified
- Knowledge base retrieval architecture
- RAG workflow

### Removed
- FAISS dependency from active workflow

### Notes
- All documents stored in local Qdrant collection
- Retrieval uses metadata tags