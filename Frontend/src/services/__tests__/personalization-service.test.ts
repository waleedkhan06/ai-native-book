/**
 * Personalization Service Tests
 * Tests for personalization features with different user profiles
 */

import { PersonalizationService } from '../personalization-service';
import { UserProfile } from '../../types/user-profile';
import { RecommendationService } from '../recommendation-service';
import { ProgressService, ProgressUpdate } from '../progress-service';

describe('Personalization Service', () => {
  test('determines content difficulty for beginner user', () => {
    const backgroundQuestions = {
      technicalBackground: 'none',
      roboticsExperience: 'no-experience',
      primaryGoal: 'learn-basics',
      hardwareAccess: 'no-hardware',
      timeCommitment: 'less-than-2-hours',
    };

    const difficulty = PersonalizationService.determineContentDifficulty(backgroundQuestions);
    expect(difficulty).toBe('beginner');
  });

  test('determines content difficulty for advanced user', () => {
    const backgroundQuestions = {
      technicalBackground: 'advanced-programming',
      roboticsExperience: 'professional-experience',
      primaryGoal: 'career-change',
      hardwareAccess: 'professional-setup',
      timeCommitment: '10-plus-hours',
    };

    const difficulty = PersonalizationService.determineContentDifficulty(backgroundQuestions);
    expect(difficulty).toBe('advanced');
  });

  test('generates appropriate learning path for project-focused user', () => {
    const backgroundQuestions = {
      technicalBackground: 'intermediate-programming',
      roboticsExperience: 'intermediate-experience',
      primaryGoal: 'build-projects',
      hardwareAccess: 'advanced-hardware',
      timeCommitment: '5-10-hours',
    };

    const learningPath = PersonalizationService.generateLearningPath(backgroundQuestions, 'build-projects');
    expect(learningPath).toBe('project-focused-hardware');
  });

  test('adapts content for beginner user', () => {
    const originalContent = 'This chapter covers ROS2 concepts.';
    const adaptedContent = PersonalizationService.adaptContent(originalContent, 'beginner', 'text');

    expect(adaptedContent).toContain('Robot Operating System 2 (ROS2)');
  });

  test('adapts content for advanced user', () => {
    const originalContent = 'This chapter covers ROS2.';
    const adaptedContent = PersonalizationService.adaptContent(originalContent, 'advanced', 'text');

    expect(adaptedContent).toContain('middleware framework');
  });
});

describe('Recommendation Service', () => {
  const mockUserProfile: UserProfile = {
    id: 'test-user',
    email: 'test@example.com',
    name: 'Test User',
    createdAt: new Date(),
    updatedAt: new Date(),
    expertiseLevel: 'beginner',
    backgroundQuestions: {
      technicalBackground: 'none',
      roboticsExperience: 'no-experience',
      primaryGoal: 'learn-basics',
      hardwareAccess: 'no-hardware',
      timeCommitment: '2-5-hours',
    },
    preferredLanguage: 'en',
    learningPath: 'foundational',
  };

  test('generates recommendations for beginner user', () => {
    const mockSettings = {
      id: 'settings-1',
      userId: 'test-user',
      currentModule: 1,
      currentChapter: 'chapter-1.1',
      progress: [],
      preferences: {
        contentDifficulty: 'adaptive',
        contentFormat: 'balanced',
        notificationFrequency: 'moderate',
        learningPace: 'moderate',
        focusArea: ['theory', 'practice'],
      },
      learningHistory: [],
      recommendedContent: [],
    };

    const recommendations = RecommendationService.generateRecommendations(mockUserProfile, mockSettings);
    expect(recommendations.length).toBeGreaterThan(0);

    // Should have beginner-appropriate recommendations
    const beginnerRec = recommendations.find(rec => rec.difficulty === 'beginner');
    expect(beginnerRec).toBeDefined();
  });

  test('filters recommendations by progress', () => {
    const mockSettings = {
      id: 'settings-1',
      userId: 'test-user',
      currentModule: 1,
      currentChapter: 'chapter-1.2',
      progress: [{
        moduleId: 1,
        completedChapters: ['chapter-1.1'],
        progressPercentage: 20,
        timeSpent: 30,
      }],
      preferences: {
        contentDifficulty: 'adaptive',
        contentFormat: 'balanced',
        notificationFrequency: 'moderate',
        learningPace: 'moderate',
        focusArea: ['theory', 'practice'],
      },
      learningHistory: [],
      recommendedContent: [],
    };

    const allRecommendations = RecommendationService.generateRecommendations(mockUserProfile, mockSettings);
    const filteredRecommendations = RecommendationService.filterRecommendationsByProgress(
      allRecommendations,
      mockSettings
    );

    // Should not recommend completed chapters
    const containsCompleted = filteredRecommendations.some(rec => rec.id === 'chapter-1.1');
    expect(containsCompleted).toBe(false);
  });
});

describe('Progress Service', () => {
  test('updates progress correctly', async () => {
    const initialSettings = {
      id: 'settings-1',
      userId: 'test-user',
      currentModule: 1,
      currentChapter: 'chapter-1.1',
      progress: [],
      preferences: {
        contentDifficulty: 'adaptive',
        contentFormat: 'balanced',
        notificationFrequency: 'moderate',
        learningPace: 'moderate',
        focusArea: ['theory', 'practice'],
      },
      learningHistory: [],
      recommendedContent: [],
    };

    const progressUpdate: ProgressUpdate = {
      moduleId: 1,
      chapterId: 'chapter-1.2',
      progressPercentage: 100,
      timeSpent: 1800, // 30 minutes in seconds
      difficultyRating: 4,
      feedback: 'Good chapter, clear explanations',
      completed: true,
    };

    const updatedSettings = await ProgressService.updateProgress(initialSettings, progressUpdate);

    expect(updatedSettings.progress).toHaveLength(1);
    expect(updatedSettings.progress[0].completedChapters).toContain('chapter-1.2');
    expect(updatedSettings.learningHistory).toHaveLength(1);
    expect(updatedSettings.learningHistory[0].chapterId).toBe('chapter-1.2');
  });

  test('calculates progress analytics', () => {
    const settingsWithProgress = {
      id: 'settings-1',
      userId: 'test-user',
      currentModule: 1,
      currentChapter: 'chapter-1.3',
      progress: [
        {
          moduleId: 1,
          completedChapters: ['chapter-1.1', 'chapter-1.2', 'chapter-1.3'],
          progressPercentage: 60,
          timeSpent: 90, // 90 minutes
        }
      ],
      preferences: {
        contentDifficulty: 'adaptive',
        contentFormat: 'balanced',
        notificationFrequency: 'moderate',
        learningPace: 'moderate',
        focusArea: ['theory', 'practice'],
      },
      learningHistory: [
        {
          chapterId: 'chapter-1.1',
          completedAt: new Date('2023-01-01'),
          timeSpent: 30,
          difficultyRating: 4,
          feedback: 'Good intro',
        },
        {
          chapterId: 'chapter-1.2',
          completedAt: new Date('2023-01-02'),
          timeSpent: 30,
          difficultyRating: 5,
          feedback: 'Very helpful',
        },
        {
          chapterId: 'chapter-1.3',
          completedAt: new Date('2023-01-03'),
          timeSpent: 30,
          difficultyRating: 3,
          feedback: 'Ok content',
        }
      ],
      recommendedContent: [],
    };

    const analytics = ProgressService.getProgressAnalytics(settingsWithProgress);

    expect(analytics.overallProgress).toBe(60);
    expect(analytics.chaptersCompleted).toBe(3);
    expect(analytics.timeSpent).toBe(90);
  });

  test('gets next chapter correctly', () => {
    const settings = {
      id: 'settings-1',
      userId: 'test-user',
      currentModule: 1,
      currentChapter: 'chapter-1.2',
      progress: [
        {
          moduleId: 1,
          completedChapters: ['module-1/chapter-1.1', 'module-1/chapter-1.2'],
          progressPercentage: 40,
          timeSpent: 60,
        }
      ],
      preferences: {
        contentDifficulty: 'adaptive',
        contentFormat: 'balanced',
        notificationFrequency: 'moderate',
        learningPace: 'moderate',
        focusArea: ['theory', 'practice'],
      },
      learningHistory: [],
      recommendedContent: [],
    };

    const nextChapter = ProgressService.getNextChapter(settings, 1);
    expect(nextChapter).toBe('module-1/chapter-1.3');
  });
});

// Integration test: Complete personalization flow
describe('Personalization Integration', () => {
  test('complete flow from user profile to recommendations', () => {
    const userProfile: UserProfile = {
      id: 'test-user-2',
      email: 'test2@example.com',
      name: 'Test User 2',
      createdAt: new Date(),
      updatedAt: new Date(),
      expertiseLevel: 'intermediate', // Will be calculated
      backgroundQuestions: {
        technicalBackground: 'intermediate-programming',
        roboticsExperience: 'intermediate-experience',
        primaryGoal: 'build-projects',
        hardwareAccess: 'basic-hardware',
        timeCommitment: '5-10-hours',
      },
      preferredLanguage: 'en',
      learningPath: 'project-focused-simulation', // Will be calculated
    };

    // Update profile with calculated values
    const updatedProfile = PersonalizationService.updateUserProfileWithCalculations(userProfile);

    expect(updatedProfile.expertiseLevel).toBe('intermediate');
    expect(updatedProfile.learningPath).toBe('project-focused-simulation');

    // Generate recommendations based on updated profile
    const settings = {
      id: 'settings-2',
      userId: 'test-user-2',
      currentModule: 1,
      currentChapter: 'chapter-1.1',
      progress: [],
      preferences: {
        contentDifficulty: 'adaptive',
        contentFormat: 'balanced',
        notificationFrequency: 'moderate',
        learningPace: 'moderate',
        focusArea: ['theory', 'practice'],
      },
      learningHistory: [],
      recommendedContent: [],
    };

    const recommendations = RecommendationService.generateRecommendations(updatedProfile, settings);
    expect(recommendations.length).toBeGreaterThan(0);

    // Should have recommendations that match the user's profile
    const projectRecommendations = recommendations.filter(rec => rec.title.toLowerCase().includes('project'));
    expect(projectRecommendations.length).toBeGreaterThan(0);
  });
});