from fastapi import FastAPI
from app.routes import chat, health

app = FastAPI(
    title="RAG Chatbot API for Physical AI & Humanoid Robotics Textbook",
    description="API for the Retrieval Augmented Generation chatbot that answers questions about the textbook content",
    version="1.0.0"
)

# Include routers
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(health.router, prefix="/health", tags=["health"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)