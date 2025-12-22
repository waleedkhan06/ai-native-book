/**
 * Recommendation Service
 * Implements content recommendation algorithm based on user profile
 */

import { UserProfile } from '../types/user-profile';
import { PersonalizationSettings } from '../types/personalization-settings';

export interface RecommendedItem {
  id: string;
  title: string;
  type: 'chapter' | 'project' | 'resource' | 'video' | 'practice';
  priority: 'high' | 'medium' | 'low';
  reason: string;
  estimatedTime: number; // in minutes
  prerequisite?: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
}

export class RecommendationService {
  /**
   * Generates content recommendations based on user profile and learning progress
   */
  static generateRecommendations(
    userProfile: UserProfile,
    settings: PersonalizationSettings
  ): RecommendedItem[] {
    const recommendations: RecommendedItem[] = [];

    // Based on expertise level
    const expertiseBasedRecs = this.getRecommendationsByExpertise(
      userProfile.expertiseLevel,
      userProfile.backgroundQuestions
    );
    recommendations.push(...expertiseBasedRecs);

    // Based on learning path
    const pathBasedRecs = this.getRecommendationsByLearningPath(
      userProfile.learningPath,
      userProfile.backgroundQuestions
    );
    recommendations.push(...pathBasedRecs);

    // Based on progress and learning history
    const progressBasedRecs = this.getRecommendationsByProgress(
      settings,
      userProfile.backgroundQuestions
    );
    recommendations.push(...progressBasedRecs);

    // Remove duplicates and sort by priority
    const uniqueRecommendations = this.removeDuplicateRecommendations(recommendations);
    return this.sortRecommendationsByPriority(uniqueRecommendations);
  }

  private static getRecommendationsByExpertise(
    expertiseLevel: 'beginner' | 'intermediate' | 'advanced',
    backgroundQuestions: any
  ): RecommendedItem[] {
    const recommendations: RecommendedItem[] = [];

    switch (expertiseLevel) {
      case 'beginner':
        recommendations.push({
          id: 'module-1/chapter-1.1',
          title: 'Introduction to Physical AI',
          type: 'chapter',
          priority: 'high',
          reason: 'Foundational concepts for beginners',
          estimatedTime: 30,
          difficulty: 'beginner'
        });
        recommendations.push({
          id: 'module-1/chapter-1.2',
          title: 'ROS2 Architecture & Nodes',
          type: 'chapter',
          priority: 'high',
          reason: 'Essential ROS2 concepts for beginners',
          estimatedTime: 45,
          difficulty: 'beginner'
        });
        recommendations.push({
          id: 'resource/ros-tutorials',
          title: 'ROS2 Beginner Tutorials',
          type: 'resource',
          priority: 'medium',
          reason: 'Additional practice for ROS2 concepts',
          estimatedTime: 60,
          difficulty: 'beginner'
        });
        break;

      case 'intermediate':
        recommendations.push({
          id: 'module-2/chapter-2.1',
          title: 'Gazebo Fundamentals',
          type: 'chapter',
          priority: 'high',
          reason: 'Building on your intermediate knowledge',
          estimatedTime: 40,
          difficulty: 'intermediate'
        });
        recommendations.push({
          id: 'module-1/chapter-1.4',
          title: 'URDF for Humanoids',
          type: 'chapter',
          priority: 'medium',
          reason: 'Expanding your robotics knowledge',
          estimatedTime: 50,
          difficulty: 'intermediate'
        });
        break;

      case 'advanced':
        recommendations.push({
          id: 'module-4/chapter-4.1',
          title: 'Vision-Language Models',
          type: 'chapter',
          priority: 'high',
          reason: 'Advanced topic matching your expertise',
          estimatedTime: 60,
          difficulty: 'advanced'
        });
        recommendations.push({
          id: 'project/advanced-vla',
          title: 'Vision-Language-Action System Project',
          type: 'project',
          priority: 'high',
          reason: 'Challenging project for advanced users',
          estimatedTime: 180,
          difficulty: 'advanced'
        });
        break;
    }

    return recommendations;
  }

  private static getRecommendationsByLearningPath(
    learningPath: string,
    backgroundQuestions: any
  ): RecommendedItem[] {
    const recommendations: RecommendedItem[] = [];

    switch (learningPath) {
      case 'project-focused-simulation':
        recommendations.push({
          id: 'module-2/chapter-2.1',
          title: 'Gazebo Fundamentals',
          type: 'chapter',
          priority: 'high',
          reason: 'Core simulation knowledge for your project-focused path',
          estimatedTime: 40
        });
        recommendations.push({
          id: 'project/simulation-project',
          title: 'Gazebo Simulation Project',
          type: 'project',
          priority: 'high',
          reason: 'Hands-on project for simulation learning',
          estimatedTime: 120
        });
        break;

      case 'project-focused-hardware':
        if (backgroundQuestions.hardwareAccess !== 'no-hardware') {
          recommendations.push({
            id: 'hardware/jetson-setup',
            title: 'Jetson Setup Guide',
            type: 'chapter',
            priority: 'high',
            reason: 'Hardware setup for your hardware-focused path',
            estimatedTime: 30
          });
          recommendations.push({
            id: 'project/hardware-project',
            title: 'Hardware Implementation Project',
            type: 'project',
            priority: 'high',
            reason: 'Practical hardware project',
            estimatedTime: 150
          });
        }
        break;

      case 'career-transition':
        recommendations.push({
          id: 'resource/career-guide',
          title: 'Robotics Career Guide',
          type: 'resource',
          priority: 'high',
          reason: 'Career-focused resources for your transition',
          estimatedTime: 20
        });
        recommendations.push({
          id: 'module-4/chapter-4.4',
          title: 'Capstone Project',
          type: 'chapter',
          priority: 'medium',
          reason: 'Comprehensive project for portfolio building',
          estimatedTime: 120
        });
        break;

      case 'academic':
        recommendations.push({
          id: 'resource/research-papers',
          title: 'Academic Research Papers',
          type: 'resource',
          priority: 'high',
          reason: 'Academic resources for deep theoretical understanding',
          estimatedTime: 90
        });
        recommendations.push({
          id: 'module-3/chapter-3.2',
          title: 'Perception Systems',
          type: 'chapter',
          priority: 'high',
          reason: 'Advanced perception concepts for academic learning',
          estimatedTime: 60
        });
        break;

      default:
        // General path recommendations
        recommendations.push({
          id: 'module-1/chapter-1.5',
          title: 'Launch Files & Parameters',
          type: 'chapter',
          priority: 'medium',
          reason: 'Important foundational concept',
          estimatedTime: 35
        });
    }

    return recommendations;
  }

  private static getRecommendationsByProgress(
    settings: PersonalizationSettings,
    backgroundQuestions: any
  ): RecommendedItem[] {
    const recommendations: RecommendedItem[] = [];

    // If user is progressing well, suggest more challenging content
    const avgProgress = settings.progress.reduce((sum, module) => sum + module.progressPercentage, 0) /
                        (settings.progress.length || 1);

    if (avgProgress > 75) {
      // User is doing well, suggest advanced content
      recommendations.push({
        id: 'challenge/advanced-concepts',
        title: 'Advanced Robotics Concepts',
        type: 'chapter',
        priority: 'medium',
        reason: 'You\'re progressing well! Try more advanced concepts',
        estimatedTime: 45
      });
    } else if (avgProgress < 40) {
      // User might need more foundational content
      recommendations.push({
        id: 'review/foundational-concepts',
        title: 'Review: Foundational Concepts',
        type: 'chapter',
        priority: 'high',
        reason: 'Reviewing foundational concepts might help',
        estimatedTime: 30
      });
    }

    // Based on time commitment
    if (backgroundQuestions.timeCommitment === '10-plus-hours') {
      recommendations.push({
        id: 'intensive/weekend-project',
        title: 'Weekend Intensive Project',
        type: 'project',
        priority: 'medium',
        reason: 'You have time for intensive projects',
        estimatedTime: 180
      });
    } else if (backgroundQuestions.timeCommitment === 'less-than-2-hours') {
      recommendations.push({
        id: 'micro/quick-lessons',
        title: 'Quick 10-Minute Lessons',
        type: 'chapter',
        priority: 'high',
        reason: 'Short lessons that fit your schedule',
        estimatedTime: 10
      });
    }

    // Based on learning pace preference
    if (settings.preferences?.learningPace === 'fast') {
      recommendations.push({
        id: 'accelerated/track',
        title: 'Accelerated Learning Track',
        type: 'resource',
        priority: 'medium',
        reason: 'Accelerated content for fast learners',
        estimatedTime: 25
      });
    }

    return recommendations;
  }

  private static removeDuplicateRecommendations(recommendations: RecommendedItem[]): RecommendedItem[] {
    const seen = new Set<string>();
    return recommendations.filter(item => {
      if (seen.has(item.id)) {
        return false;
      }
      seen.add(item.id);
      return true;
    });
  }

  private static sortRecommendationsByPriority(recommendations: RecommendedItem[]): RecommendedItem[] {
    const priorityOrder = { high: 3, medium: 2, low: 1 };

    return recommendations.sort((a, b) => {
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  /**
   * Filters recommendations based on user's current progress to avoid suggesting completed content
   */
  static filterRecommendationsByProgress(
    recommendations: RecommendedItem[],
    settings: PersonalizationSettings
  ): RecommendedItem[] {
    const completedChapters = settings.progress.flatMap(module => module.completedChapters);

    return recommendations.filter(rec => {
      // Don't recommend chapters that are already completed
      return !completedChapters.includes(rec.id);
    });
  }

  /**
   * Gets next recommended chapter based on user's current position and progress
   */
  static getNextChapterRecommendation(
    userProfile: UserProfile,
    settings: PersonalizationSettings
  ): RecommendedItem | null {
    const allRecommendations = this.generateRecommendations(userProfile, settings);
    const filteredRecommendations = this.filterRecommendationsByProgress(allRecommendations, settings);

    // Find the first high-priority recommendation
    const nextChapter = filteredRecommendations.find(rec =>
      rec.type === 'chapter' && rec.priority === 'high'
    );

    if (nextChapter) {
      return nextChapter;
    }

    // If no high-priority chapter, return the first available
    return filteredRecommendations.find(rec => rec.type === 'chapter') ||
           filteredRecommendations[0] || null;
  }
}