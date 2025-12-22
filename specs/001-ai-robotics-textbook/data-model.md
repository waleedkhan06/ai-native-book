# Data Model: Physical AI & Humanoid Robotics Textbook

## Phase 1: Entity Definitions

### User Entity
- **id**: Unique identifier for each user
- **email**: User's email address (required, unique)
- **name**: User's full name
- **createdAt**: Timestamp of account creation
- **updatedAt**: Timestamp of last update
- **expertiseLevel**: Technical expertise level (beginner, intermediate, advanced)
- **backgroundQuestions**: JSON object containing responses to 5 background questions
- **preferredLanguage**: Default language preference ('en' or 'ur')
- **learningPath**: Personalized learning path based on questionnaire

### Chapter Entity
- **id**: Unique identifier for each chapter
- **title**: Chapter title
- **content**: Chapter content in Markdown format
- **module**: Module number (1-4) this chapter belongs to
- **order**: Chapter number within the module
- **difficulty**: Difficulty level (1-5 scale)
- **duration**: Estimated reading time in minutes
- **prerequisites**: List of prerequisite chapters
- **codeExamples**: Array of code examples in the chapter
- **language**: Content language ('en' or 'ur')

### Module Entity
- **id**: Unique identifier for each module
- **title**: Module title
- **description**: Brief description of the module
- **order**: Module number (1-4)
- **totalChapters**: Number of chapters in the module
- **duration**: Estimated total time to complete module
- **topics**: List of topics covered in the module

### ChatbotQuery Entity
- **id**: Unique identifier for each query
- **userId**: Reference to the user who made the query
- **queryText**: Original text of the user's question
- **queryEmbedding**: Vector embedding of the query text
- **context**: Context from textbook content used to answer
- **response**: The chatbot's response to the query
- **timestamp**: When the query was made
- **accuracy**: Confidence score of the response
- **feedback**: User feedback on response quality

### PersonalizationSettings Entity
- **id**: Unique identifier for personalization settings
- **userId**: Reference to the user
- **currentModule**: Currently active module
- **currentChapter**: Currently active chapter
- **progress**: Completion percentage for each module
- **preferences**: User preferences for content delivery
- **learningHistory**: History of completed chapters and assessments
- **recommendedContent**: Personalized content recommendations

### Translation Entity
- **id**: Unique identifier for translation pair
- **contentId**: Reference to original content (chapter, module, etc.)
- **language**: Target language ('ur')
- **translatedContent**: Translated content text
- **status**: Translation status (pending, reviewed, approved)
- **lastUpdated**: Timestamp of last translation update

## Relationships

### User → Chapter
- **UserProgress**: Tracks user's progress through chapters
- **User has many UserProgress records**
- **Chapter has many UserProgress records**

### Module → Chapter
- **Module contains many Chapters**
- **Chapter belongs to one Module**

### User → ChatbotQuery
- **User makes many ChatbotQueries**
- **ChatbotQuery belongs to one User**

### User → PersonalizationSettings
- **User has one PersonalizationSettings**
- **PersonalizationSettings belongs to one User**

### Content → Translation
- **Content has many Translations**
- **Translation belongs to one Content**

## Validation Rules

### User Entity
- Email must be valid format
- Expertise level must be one of: 'beginner', 'intermediate', 'advanced'
- Background questions must contain all 5 required responses

### Chapter Entity
- Title must not be empty
- Content must be in valid Markdown format
- Module number must be between 1 and 4
- Order must be positive integer
- Difficulty must be between 1 and 5

### Module Entity
- Title must not be empty
- Order must be between 1 and 4
- Total chapters must be positive integer
- Duration must be positive number

### ChatbotQuery Entity
- Query text must not be empty
- Timestamp must be current or past
- Accuracy must be between 0 and 1

### PersonalizationSettings Entity
- Progress percentages must be between 0 and 100
- Current module and chapter must be valid references

## State Transitions

### User Registration Flow
1. User account created (unverified)
2. Background questions completed (active)
3. Personalization settings initialized (personalized)

### Content Progression
1. Chapter assigned to user (pending)
2. User starts reading (in-progress)
3. User completes chapter (completed)

### Translation Process
1. Translation requested (pending)
2. Translation completed (translated)
3. Translation reviewed (reviewed)
4. Translation approved (approved)