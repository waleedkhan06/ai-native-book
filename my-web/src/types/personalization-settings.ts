/**
 * Personalization Settings Model
 * Represents user's personalization preferences and learning progress
 */

export interface PersonalizationSettings {
  id: string;
  userId: string;
  currentModule: number;
  currentChapter: string;
  progress: ModuleProgress[];
  preferences: UserPreferences;
  learningHistory: LearningHistory[];
  recommendedContent: RecommendedContent[];
}

export interface ModuleProgress {
  moduleId: number;
  completedChapters: string[];
  progressPercentage: number;
  timeSpent: number; // in minutes
}

export interface UserPreferences {
  contentDifficulty: 'adaptive' | 'beginner' | 'intermediate' | 'advanced';
  contentFormat: 'text-heavy' | 'visual-heavy' | 'balanced';
  notificationFrequency: 'frequent' | 'moderate' | 'minimal';
  learningPace: 'fast' | 'moderate' | 'slow';
  focusArea: string[]; // e.g., ['theory', 'practice', 'projects']
}

export interface LearningHistory {
  chapterId: string;
  completedAt: Date;
  timeSpent: number; // in minutes
  difficultyRating: number; // 1-5 scale
  feedback: string;
}

export interface RecommendedContent {
  contentId: string;
  contentType: 'chapter' | 'project' | 'resource' | 'video';
  reason: string;
  priority: 'high' | 'medium' | 'low';
  estimatedTime: number; // in minutes
}

export interface PersonalizationSettingsInput {
  currentModule?: number;
  currentChapter?: string;
  progress?: Partial<ModuleProgress>[];
  preferences?: Partial<UserPreferences>;
  recommendedContent?: RecommendedContent[];
}