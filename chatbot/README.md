# RAG Chatbot for Physical AI & Humanoid Robotics Textbook

This is the backend service for the RAG (Retrieval Augmented Generation) chatbot that answers questions about the textbook content.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Cohere API key and other configuration
   ```

4. Run the development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## Endpoints

- `POST /chat/query` - Submit a question to the RAG chatbot
- `GET /chat/history/{user_id}` - Get chat history for a user
- `POST /content/embed` - Embed textbook content for RAG retrieval
- `GET /health` - Health check endpoint