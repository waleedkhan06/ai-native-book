/**
 * Progress Tracking Service
 * Manages user learning progress and analytics
 */

import { PersonalizationSettings, ModuleProgress, LearningHistory } from '../types/personalization-settings';
import { UserProfile } from '../types/user-profile';

export interface ProgressUpdate {
  moduleId: number;
  chapterId: string;
  progressPercentage: number;
  timeSpent?: number; // in seconds
  difficultyRating?: number; // 1-5 scale
  feedback?: string;
  completed?: boolean;
}

export interface ProgressAnalytics {
  overallProgress: number;
  moduleProgress: { moduleId: number; progress: number }[];
  timeSpent: number; // total time in minutes
  chaptersCompleted: number;
  currentStreak: number;
  longestStreak: number;
  avgTimePerChapter: number; // in minutes
  difficultyTrend: number[]; // array of ratings
  lastActive: Date;
}

export class ProgressService {
  /**
   * Updates user progress for a specific chapter
   */
  static async updateProgress(
    settings: PersonalizationSettings,
    update: ProgressUpdate
  ): Promise<PersonalizationSettings> {
    const { moduleId, chapterId, progressPercentage, timeSpent, difficultyRating, feedback, completed } = update;

    // Find or create module progress
    let moduleProgress = settings.progress.find(mp => mp.moduleId === moduleId);

    if (!moduleProgress) {
      moduleProgress = {
        moduleId,
        completedChapters: [],
        progressPercentage: 0,
        timeSpent: 0,
      };
      settings.progress.push(moduleProgress);
    }

    // Update completed chapters if this chapter is completed
    if (completed) {
      if (!moduleProgress.completedChapters.includes(chapterId)) {
        moduleProgress.completedChapters.push(chapterId);
      }
    }

    // Update time spent
    if (timeSpent) {
      moduleProgress.timeSpent += timeSpent / 60; // convert seconds to minutes
    }

    // Calculate module progress based on completed chapters
    // For simplicity, assuming each module has 5 chapters
    moduleProgress.progressPercentage = Math.min(100,
      Math.round((moduleProgress.completedChapters.length / 5) * 100)
    );

    // Add to learning history
    if (completed || difficultyRating || feedback) {
      const historyEntry: LearningHistory = {
        chapterId,
        completedAt: new Date(),
        timeSpent: timeSpent ? timeSpent / 60 : 0, // convert to minutes
        difficultyRating: difficultyRating || 0,
        feedback: feedback || '',
      };
      settings.learningHistory.push(historyEntry);
    }

    // Update current chapter in settings
    settings.currentChapter = chapterId;

    return settings;
  }

  /**
   * Gets progress analytics for a user
   */
  static getProgressAnalytics(settings: PersonalizationSettings): ProgressAnalytics {
    const totalModules = settings.progress.length;
    const totalChapters = settings.progress.reduce((sum, module) => sum + module.completedChapters.length, 0);
    const totalTimeSpent = settings.progress.reduce((sum, module) => sum + module.timeSpent, 0);

    // Calculate overall progress (simplified: equal weight per module)
    const overallProgress = totalModules > 0
      ? settings.progress.reduce((sum, module) => sum + module.progressPercentage, 0) / totalModules
      : 0;

    // Calculate streaks
    const { currentStreak, longestStreak } = this.calculateStreaks(settings.learningHistory);

    // Calculate average time per chapter
    const avgTimePerChapter = totalChapters > 0 ? totalTimeSpent / totalChapters : 0;

    // Get difficulty trend (last 10 ratings)
    const difficultyTrend = settings.learningHistory
      .slice(-10)
      .map(entry => entry.difficultyRating)
      .filter(rating => rating > 0);

    // Get last active date
    const lastActive = settings.learningHistory.length > 0
      ? settings.learningHistory[settings.learningHistory.length - 1].completedAt
      : new Date(0);

    return {
      overallProgress: Math.round(overallProgress),
      moduleProgress: settings.progress.map(module => ({
        moduleId: module.moduleId,
        progress: module.progressPercentage
      })),
      timeSpent: Math.round(totalTimeSpent),
      chaptersCompleted: totalChapters,
      currentStreak,
      longestStreak,
      avgTimePerChapter: Math.round(avgTimePerChapter * 100) / 100,
      difficultyTrend,
      lastActive
    };
  }

  /**
   * Calculates learning streaks based on learning history
   */
  private static calculateStreaks(learningHistory: LearningHistory[]): { currentStreak: number; longestStreak: number } {
    if (learningHistory.length === 0) {
      return { currentStreak: 0, longestStreak: 0 };
    }

    // Sort history by date
    const sortedHistory = [...learningHistory].sort((a, b) =>
      new Date(a.completedAt).getTime() - new Date(b.completedAt).getTime()
    );

    let currentStreak = 1;
    let longestStreak = 1;
    let previousDate = new Date(sortedHistory[0].completedAt);

    // Convert to dates and check consecutive days
    for (let i = 1; i < sortedHistory.length; i++) {
      const currentDate = new Date(sortedHistory[i].completedAt);

      // Calculate difference in days
      const diffTime = currentDate.getTime() - previousDate.getTime();
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays === 1) {
        // Consecutive day
        currentStreak++;
      } else if (diffDays > 1) {
        // Gap in learning - reset current streak
        longestStreak = Math.max(longestStreak, currentStreak);
        currentStreak = 1;
      }

      previousDate = currentDate;
    }

    longestStreak = Math.max(longestStreak, currentStreak);

    return { currentStreak, longestStreak };
  }

  /**
   * Gets module progress for a specific module
   */
  static getModuleProgress(settings: PersonalizationSettings, moduleId: number): ModuleProgress | null {
    return settings.progress.find(mp => mp.moduleId === moduleId) || null;
  }

  /**
   * Gets completion status for a specific chapter
   */
  static isChapterCompleted(settings: PersonalizationSettings, moduleId: number, chapterId: string): boolean {
    const moduleProgress = settings.progress.find(mp => mp.moduleId === moduleId);
    return moduleProgress ? moduleProgress.completedChapters.includes(chapterId) : false;
  }

  /**
   * Gets next chapter recommendation based on progress
   */
  static getNextChapter(settings: PersonalizationSettings, currentModuleId: number): string | null {
    const moduleProgress = settings.progress.find(mp => mp.moduleId === currentModuleId);

    if (!moduleProgress) {
      // If no progress for this module, start with first chapter
      return `module-${currentModuleId}/chapter-${currentModuleId}.1`;
    }

    // Find the next uncompleted chapter in the module
    // For this example, assuming chapters follow pattern: module-X/chapter-X.Y
    const completedChapters = moduleProgress.completedChapters;

    // Extract chapter numbers and find the next one
    const completedNumbers = completedChapters
      .filter(chapter => chapter.startsWith(`module-${currentModuleId}/chapter-${currentModuleId}.`))
      .map(chapter => {
        const match = chapter.match(/chapter-\d+\.(\d+)/);
        return match ? parseInt(match[1]) : 0;
      })
      .filter(num => !isNaN(num))
      .sort((a, b) => a - b);

    if (completedNumbers.length === 0) {
      return `module-${currentModuleId}/chapter-${currentModuleId}.1`;
    }

    // Find the first missing chapter number
    for (let i = 1; i <= 10; i++) { // Assuming max 10 chapters per module
      if (!completedNumbers.includes(i)) {
        return `module-${currentModuleId}/chapter-${currentModuleId}.${i}`;
      }
    }

    return null; // No more chapters in this module
  }

  /**
   * Calculates time to completion based on progress and learning pace
   */
  static getTimeToCompletion(
    settings: PersonalizationSettings,
    totalChapters: number,
    userProfile: UserProfile
  ): { estimatedDays: number; chaptersRemaining: number } {
    const completedChapters = settings.progress.reduce(
      (sum, module) => sum + module.completedChapters.length, 0
    );
    const chaptersRemaining = totalChapters - completedChapters;

    if (chaptersRemaining <= 0) {
      return { estimatedDays: 0, chaptersRemaining: 0 };
    }

    // Determine chapters per day based on user's time commitment and expertise
    let chaptersPerDay = 1; // default

    switch (userProfile.backgroundQuestions.timeCommitment) {
      case '10-plus-hours':
        chaptersPerDay = userProfile.expertiseLevel === 'advanced' ? 3 : 2;
        break;
      case '5-10-hours':
        chaptersPerDay = userProfile.expertiseLevel === 'advanced' ? 2 : 1.5;
        break;
      case '2-5-hours':
        chaptersPerDay = 1;
        break;
      case 'less-than-2-hours':
        chaptersPerDay = 0.5; // half chapter per day
        break;
    }

    const estimatedDays = chaptersRemaining / chaptersPerDay;

    return {
      estimatedDays: Math.ceil(estimatedDays),
      chaptersRemaining
    };
  }

  /**
   * Resets progress for a specific module
   */
  static resetModuleProgress(settings: PersonalizationSettings, moduleId: number): PersonalizationSettings {
    const moduleProgressIndex = settings.progress.findIndex(mp => mp.moduleId === moduleId);

    if (moduleProgressIndex !== -1) {
      settings.progress.splice(moduleProgressIndex, 1);
    }

    // Also remove related learning history
    settings.learningHistory = settings.learningHistory.filter(
      entry => !entry.chapterId.startsWith(`module-${moduleId}/`)
    );

    return settings;
  }
}