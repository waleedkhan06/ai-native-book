/**
 * User Profile Model
 * Represents user profile data with expertise level and background questions
 */

export interface UserProfile {
  id: string;
  email: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced';
  backgroundQuestions: BackgroundQuestions;
  preferredLanguage: 'en' | 'ur';
  learningPath: string;
}

export interface BackgroundQuestions {
  /**
   * Question 1: What is your technical background?
   * Options: 'none', 'basic-programming', 'intermediate-programming', 'advanced-programming', 'robotics-experience'
   */
  technicalBackground: string;

  /**
   * Question 2: What is your robotics experience?
   * Options: 'no-experience', 'basic-experience', 'intermediate-experience', 'advanced-experience', 'professional-experience'
   */
  roboticsExperience: string;

  /**
   * Question 3: What is your primary goal?
   * Options: 'learn-basics', 'build-projects', 'career-change', 'academic-pursuit', 'hobby-interest'
   */
  primaryGoal: string;

  /**
   * Question 4: What hardware do you have access to?
   * Options: 'no-hardware', 'simulator-only', 'basic-hardware', 'advanced-hardware', 'professional-setup'
   */
  hardwareAccess: string;

  /**
   * Question 5: How much time can you dedicate weekly?
   * Options: 'less-than-2-hours', '2-5-hours', '5-10-hours', '10-plus-hours'
   */
  timeCommitment: string;
}

export interface UserProfileInput {
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced';
  backgroundQuestions: BackgroundQuestions;
  preferredLanguage?: 'en' | 'ur';
  learningPath?: string;
}