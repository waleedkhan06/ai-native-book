# Quickstart Guide: RAG Chatbot System

## Overview
This guide will help you quickly set up and start using the RAG (Retrieval-Augmented Generation) Chatbot system. The system integrates with Qdrant for vector storage, Neon Postgres for metadata, and OpenRouter for LLM access.

## Prerequisites
- Python 3.11+
- pip package manager
- Git (for cloning the repository)
- Access to the following services:
  - Qdrant Cloud account
  - Neon Postgres account
  - OpenRouter account

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag-chatbot-backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration

### 1. Environment Variables
Create a `.env` file in the project root with the following variables:

```env
# Qdrant Configuration
QDRANT_URL=https://5ae53cc1-dbc4-44ef-a5d9-9a27778a70f9.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.FAwrR6glYWhLMLpx_yR4gOGN4nkRtiuNMJGFWIYe3EM
QDRANT_CLUSTER_ID=5ae53cc1-dbc4-44ef-a5d9-9a27778a70f9

# Neon Postgres Configuration
DATABASE_URL=postgresql://neondb_owner:npg_DTLjYQkl12UB@ep-dark-dew-ahg8v6x1-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

# OpenRouter Configuration
OPENROUTER_API_KEY=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

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=True
SECRET_KEY=your-secret-key-here
```

**⚠️ Security Warning**: Never commit these credentials to version control. The `.env` file should be included in your `.gitignore`.

### 2. Initialize the Database
```bash
python -m src.database.init
```

### 3. Configure Qdrant Collection
```bash
python -m src.vector_store.setup_collection
```

## Running the Application

### 1. Start the Development Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Verify the Service is Running
Open your browser and navigate to `http://localhost:8000/health` to verify the service is operational.

## Making Your First Request

### 1. Using curl
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What are the key principles of RAG systems?"
      }
    ]
  }'
```

### 2. Using Python Requests
```python
import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-jwt-token"
}

data = {
    "messages": [
        {
            "role": "user",
            "content": "What are the key principles of RAG systems?"
        }
    ]
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=data,
    headers=headers
)

print(response.json())
```

## Adding Documents to the Knowledge Base

### 1. Using the API
```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "title": "My Document",
    "content": "Full content of the document goes here...",
    "source_url": "https://example.com/document.pdf"
  }'
```

### 2. Batch Processing Script
```bash
python -m src.ingestion.batch_ingest --directory /path/to/documents
```

## Testing the Application

### 1. Run Unit Tests
```bash
pytest tests/unit/
```

### 2. Run Integration Tests
```bash
pytest tests/integration/
```

### 3. Run Contract Tests
```bash
pytest tests/contract/
```

## Troubleshooting

### Common Issues

#### Issue: Connection to Qdrant fails
**Solution**: Verify your QDRANT_URL and QDRANT_API_KEY in the environment variables. Check that your IP is allowed to connect to the Qdrant cluster.

#### Issue: Database connection fails
**Solution**: Verify your DATABASE_URL is correct and that you can connect to the Neon Postgres instance directly.

#### Issue: Rate limiting on OpenRouter
**Solution**: The system has built-in fallback mechanisms. Check your OpenRouter account limits and billing status.

#### Issue: Embedding model not working
**Solution**: Ensure the embedding model specified in your configuration is available on OpenRouter.

### Enable Debug Logging
Set `LOG_LEVEL=DEBUG` in your environment variables to get more detailed logs.

## Next Steps

1. **Customize your knowledge base**: Add your own documents to the RAG system
2. **Fine-tune parameters**: Adjust temperature, max_tokens, and other parameters for your use case
3. **Monitor performance**: Set up logging and monitoring for production use
4. **Secure your deployment**: Implement proper authentication and authorization for production
5. **Scale the system**: Configure load balancing and horizontal scaling as needed

## Additional Resources

- [API Documentation](./contracts/chatbot-api.yaml)
- [Data Model](./data-model.md)
- [Research Summary](./research.md)
- [Full Implementation Plan](./plan.md)