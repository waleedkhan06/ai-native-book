# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Getting Started

This guide will help you set up and run the Physical AI & Humanoid Robotics textbook platform locally.

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.11+
- Docker and Docker Compose (for containerized services)
- Ubuntu 22.04 (recommended) or equivalent Linux distribution

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Set up the Docusaurus frontend**
   ```bash
   cd my-web
   npm install
   ```

3. **Set up the chatbot backend**
   ```bash
   cd chatbot
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # In the chatbot directory
   cp .env.example .env
   # Edit .env with your Cohere API key and other configuration
   ```

5. **Start the services**
   ```bash
   # Terminal 1: Start Docusaurus development server
   cd my-web
   npm start

   # Terminal 2: Start chatbot backend
   cd chatbot
   python -m uvicorn app.main:app --reload --port 8000
   ```

6. **Initialize the vector database (Qdrant)**
   ```bash
   # In a third terminal
   docker run -d --name qdrant-container -p 6333:6333 qdrant/qdrant
   ```

### Development Workflow

1. **Adding textbook content**
   - Add new chapters to the `my-web/docs/module-X/` directories
   - Update `my-web/sidebars.ts` to include new content in navigation
   - Ensure all code examples are tested on Ubuntu 22.04

2. **Working with the RAG chatbot**
   - Add new textbook content to the vector database using the ingestion script
   - Test chatbot responses via the API at `http://localhost:8000/chat`
   - Monitor accuracy metrics to ensure >90% performance

3. **Testing personalization**
   - Complete the 5 background questions during signup
   - Verify content adapts based on your expertise level
   - Check that learning path recommendations are appropriate

4. **Testing Urdu translation**
   - Use the language toggle in the UI
   - Verify all content appears in Urdu when selected
   - Check that navigation and UI elements are properly translated

### Running Tests

1. **Frontend tests**
   ```bash
   cd my-web
   npm test
   ```

2. **Backend tests**
   ```bash
   cd chatbot
   pytest
   ```

3. **End-to-end tests**
   ```bash
   # After starting all services
   cd my-web
   npx playwright test
   ```

### Deployment

1. **Build the Docusaurus site**
   ```bash
   cd my-web
   npm run build
   ```

2. **Deploy to GitHub Pages**
   ```bash
   npm run deploy
   ```

3. **Deploy the chatbot backend** to your preferred hosting platform (AWS, GCP, etc.)

### Troubleshooting

- **Docusaurus won't start**: Ensure you're in the `my-web` directory and have run `npm install`
- **Chatbot API not responding**: Check that the backend is running on port 8000 and environment variables are set
- **Translation not working**: Verify that Urdu content has been properly added to the translation database
- **Performance issues**: Check that you're meeting the Lighthouse >95 requirement and chatbot response time <2s

### Next Steps

1. Review the complete implementation plan in `specs/001-ai-robotics-textbook/plan.md`
2. Check the detailed tasks in `specs/001-ai-robotics-textbook/tasks.md` (generated in next phase)
3. Explore the API contracts in `specs/001-ai-robotics-textbook/contracts/`