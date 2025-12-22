# Research Findings: Physical AI & Humanoid Robotics Textbook

## Phase 0: Research and Unknown Resolution

### Decision: Docusaurus Configuration Update
**Rationale**: Using existing Docusaurus project in 'my-web/' directory provides a solid foundation that meets constitutional requirements for Docusaurus 3.9 with GitHub Pages deployment. This approach leverages existing setup while allowing for customization needed for the textbook platform.

**Alternatives considered**:
- Building from scratch: More time-consuming and reinventing existing solutions
- Using alternative documentation platforms: Would not meet constitutional requirement for Docusaurus
- Separate repository: Would complicate deployment and maintenance

### Decision: FastAPI for RAG Chatbot Backend
**Rationale**: FastAPI aligns with constitutional requirement (IV) for RAG chatbot implementation. It provides excellent async support, automatic API documentation, and strong integration capabilities with Cohere and Qdrant needed for >90% accuracy target.

**Alternatives considered**:
- Flask: Less performant and fewer built-in features
- Node.js/Express: Would require additional complexity for Python-based AI libraries
- Direct Cohere API integration: Would lack proper abstraction and scalability

### Decision: Qdrant for Vector Database
**Rationale**: Qdrant provides excellent performance for similarity search needed in RAG system, has good Python SDK, and integrates well with Cohere embeddings. Meets accuracy requirements while being efficient for textbook content retrieval.

**Alternatives considered**:
- Pinecone: Cloud-only, less control over data
- Weaviate: More complex setup
- PostgreSQL with pgvector: Less optimized for vector similarity search

### Decision: Better Auth for User Management
**Rationale**: Better Auth meets constitutional requirement (V) for background questions during signup and provides the personalization capabilities required by constitutional principle (VI). It's lightweight and designed for modern web applications.

**Alternatives considered**:
- Auth0: More complex and costly for this use case
- Firebase Auth: Would require additional setup for custom questionnaire
- Custom auth solution: Would be reinventing existing solution

### Decision: Urdu Translation Implementation
**Rationale**: Docusaurus has built-in i18n support that can be leveraged for Urdu translation toggle, meeting constitutional requirement (VII) efficiently without significant additional complexity.

**Alternatives considered**:
- Third-party translation services: Less control over quality
- Manual separate versions: Would be harder to maintain consistency
- No translation: Would not meet constitutional requirement

### Decision: Content Structure Organization
**Rationale**: Organizing content into 4 modules (17 chapters) as specified in requirements allows for logical progression from fundamentals to advanced topics. The additional hardware guide and projects sections provide practical application opportunities.

**Alternatives considered**:
- Different module organization: Would not align with specified requirements
- Flatter structure: Would make navigation more difficult for users
- More/less modules: Would not match specified 4-module, 17-chapter requirement