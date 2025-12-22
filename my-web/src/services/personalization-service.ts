/**
 * Personalization Service
 * Manages user preferences and content personalization
 */

import { UserProfile, BackgroundQuestions } from '../types/user-profile';
import { PersonalizationSettings } from '../types/personalization-settings';

export class PersonalizationService {
  /**
   * Determines content difficulty level based on user's background questions
   */
  static determineContentDifficulty(backgroundQuestions: BackgroundQuestions): 'beginner' | 'intermediate' | 'advanced' {
    // Weighted scoring system based on background questions
    let score = 0;

    // Technical background scoring
    switch (backgroundQuestions.technicalBackground) {
      case 'none':
        score += 0;
        break;
      case 'basic-programming':
        score += 1;
        break;
      case 'intermediate-programming':
        score += 2;
        break;
      case 'advanced-programming':
        score += 3;
        break;
      case 'robotics-experience':
        score += 3;
        break;
    }

    // Robotics experience scoring
    switch (backgroundQuestions.roboticsExperience) {
      case 'no-experience':
        score += 0;
        break;
      case 'basic-experience':
        score += 1;
        break;
      case 'intermediate-experience':
        score += 2;
        break;
      case 'advanced-experience':
        score += 3;
        break;
      case 'professional-experience':
        score += 4;
        break;
    }

    // Time commitment scoring
    switch (backgroundQuestions.timeCommitment) {
      case 'less-than-2-hours':
        score += 0;
        break;
      case '2-5-hours':
        score += 1;
        break;
      case '5-10-hours':
        score += 2;
        break;
      case '10-plus-hours':
        score += 3;
        break;
    }

    // Calculate average score and determine difficulty level
    const averageScore = score / 3; // 3 factors considered

    if (averageScore >= 2.5) {
      return 'advanced';
    } else if (averageScore >= 1.5) {
      return 'intermediate';
    } else {
      return 'beginner';
    }
  }

  /**
   * Generates personalized learning path based on user profile
   */
  static generateLearningPath(backgroundQuestions: BackgroundQuestions, primaryGoal: string): string {
    // Generate learning path based on primary goal and background
    if (primaryGoal === 'career-change') {
      return 'career-transition';
    } else if (primaryGoal === 'academic-pursuit') {
      return 'academic';
    } else if (primaryGoal === 'build-projects') {
      if (backgroundQuestions.hardwareAccess !== 'no-hardware') {
        return 'project-focused-hardware';
      } else {
        return 'project-focused-simulation';
      }
    } else if (primaryGoal === 'learn-basics') {
      return 'foundational';
    } else {
      return 'general';
    }
  }

  /**
   * Adjusts content based on user expertise level
   */
  static adaptContent(content: string, userExpertise: 'beginner' | 'intermediate' | 'advanced', contentType: 'text' | 'code' | 'diagram' = 'text'): string {
    if (userExpertise === 'beginner') {
      // For beginners, add more explanations and context
      if (contentType === 'code') {
        return this.addBeginnerExplanations(content);
      } else {
        return this.simplifyContent(content);
      }
    } else if (userExpertise === 'advanced') {
      // For advanced users, provide more concise and technical content
      return this.addTechnicalDepth(content);
    } else {
      // For intermediate users, provide balanced content
      return content;
    }
  }

  private static addBeginnerExplanations(content: string): string {
    // Add explanations for complex code concepts
    return content
      .replace(/(?<!\w)ROS2(?!\w)/g, 'Robot Operating System 2 (ROS2) - a flexible framework for writing robot software')
      .replace(/(?<!\w)Qdrant(?!\w)/g, 'Qdrant - a vector database for neural network search')
      .replace(/(?<!\w)Cohere(?!\w)/g, 'Cohere - an AI platform for natural language processing');
  }

  private static simplifyContent(content: string): string {
    // Simplify complex terminology and add more explanations
    return content
      .replace(/\b(abstraction|encapsulation|inheritance)\b/gi, (match) => {
        return `${match} (a programming concept that helps organize code)`;
      });
  }

  private static addTechnicalDepth(content: string): string {
    // Add more technical details and advanced concepts
    return content
      .replace(/(?<!\w)ROS2(?!\w)/g, 'ROS2 (Robot Operating System 2) - a middleware framework providing services like hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more');
  }

  /**
   * Gets recommended content based on user profile
   */
  static getRecommendedContent(userProfile: UserProfile): string[] {
    const recommendations: string[] = [];

    // Based on expertise level, recommend appropriate content
    if (userProfile.expertiseLevel === 'beginner') {
      recommendations.push('module-1/chapter-1.1', 'module-1/chapter-1.2');
    } else if (userProfile.expertiseLevel === 'intermediate') {
      recommendations.push('module-2/chapter-2.1', 'module-1/chapter-1.4');
    } else {
      recommendations.push('module-3/chapter-3.2', 'module-4/chapter-4.1');
    }

    // Based on primary goal, recommend specific content
    if (userProfile.backgroundQuestions.primaryGoal === 'build-projects') {
      recommendations.push('projects/project-1', 'hardware/guide');
    }

    return recommendations;
  }

  /**
   * Updates user profile with calculated expertise level and learning path
   */
  static updateUserProfileWithCalculations(userProfile: UserProfile): UserProfile {
    const calculatedExpertise = this.determineContentDifficulty(userProfile.backgroundQuestions);
    const calculatedLearningPath = this.generateLearningPath(
      userProfile.backgroundQuestions,
      userProfile.backgroundQuestions.primaryGoal
    );

    return {
      ...userProfile,
      expertiseLevel: calculatedExpertise,
      learningPath: calculatedLearningPath
    };
  }
}