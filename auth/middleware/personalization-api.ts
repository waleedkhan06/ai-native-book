/**
 * Personalization API Middleware
 * Handles requests for managing personalization settings
 */

import { NextApiRequest, NextApiResponse } from 'next';
import { auth } from '../better-auth.config';
import { PersonalizationSettings, PersonalizationSettingsInput } from '../types/personalization-settings';
import { UserProfile } from '../types/user-profile';

// Mock database - in production, this would connect to your actual database
const mockPersonalizationDb: Record<string, PersonalizationSettings> = {};

// Personalization API endpoints
export default async function personalizationApi(req: NextApiRequest, res: NextApiResponse) {
  // Authenticate user
  const session = await auth.api.getSession({
    headers: req.headers,
  });

  if (!session) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const userId = session.user.id;

  try {
    switch (req.method) {
      case 'GET':
        return await getPersonalizationSettings(req, res, userId);
      case 'POST':
        return await updatePersonalizationSettings(req, res, userId);
      case 'PUT':
        return await updatePersonalizationSettings(req, res, userId);
      case 'PATCH':
        return await updatePartialPersonalizationSettings(req, res, userId);
      default:
        res.setHeader('Allow', ['GET', 'POST', 'PUT', 'PATCH']);
        return res.status(405).json({ error: `Method ${req.method} Not Allowed` });
    }
  } catch (error) {
    console.error('Personalization API Error:', error);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}

// GET /api/personalization/settings
async function getPersonalizationSettings(req: NextApiRequest, res: NextApiResponse, userId: string) {
  try {
    // Check if user has existing personalization settings
    if (mockPersonalizationDb[userId]) {
      return res.status(200).json(mockPersonalizationDb[userId]);
    }

    // If no settings exist, create default settings
    const defaultSettings: PersonalizationSettings = {
      id: `settings_${Date.now()}`,
      userId,
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

    mockPersonalizationDb[userId] = defaultSettings;
    return res.status(200).json(defaultSettings);
  } catch (error) {
    console.error('Error getting personalization settings:', error);
    return res.status(500).json({ error: 'Failed to retrieve personalization settings' });
  }
}

// POST/PUT /api/personalization/settings
async function updatePersonalizationSettings(req: NextApiRequest, res: NextApiResponse, userId: string) {
  try {
    const settingsInput: PersonalizationSettingsInput = req.body;

    // Validate input
    if (!validatePersonalizationSettingsInput(settingsInput)) {
      return res.status(400).json({ error: 'Invalid personalization settings data' });
    }

    // Create or update settings
    const settings: PersonalizationSettings = {
      id: mockPersonalizationDb[userId]?.id || `settings_${Date.now()}`,
      userId,
      currentModule: settingsInput.currentModule || mockPersonalizationDb[userId]?.currentModule || 1,
      currentChapter: settingsInput.currentChapter || mockPersonalizationDb[userId]?.currentChapter || 'chapter-1.1',
      progress: settingsInput.progress || mockPersonalizationDb[userId]?.progress || [],
      preferences: {
        ...mockPersonalizationDb[userId]?.preferences,
        ...settingsInput.preferences,
      },
      learningHistory: settingsInput.learningHistory || mockPersonalizationDb[userId]?.learningHistory || [],
      recommendedContent: settingsInput.recommendedContent || mockPersonalizationDb[userId]?.recommendedContent || [],
    };

    mockPersonalizationDb[userId] = settings;
    return res.status(200).json(settings);
  } catch (error) {
    console.error('Error updating personalization settings:', error);
    return res.status(500).json({ error: 'Failed to update personalization settings' });
  }
}

// PATCH /api/personalization/settings
async function updatePartialPersonalizationSettings(req: NextApiRequest, res: NextApiResponse, userId: string) {
  try {
    const settingsInput: Partial<PersonalizationSettingsInput> = req.body;

    // Get existing settings or create defaults
    let existingSettings = mockPersonalizationDb[userId];
    if (!existingSettings) {
      existingSettings = {
        id: `settings_${Date.now()}`,
        userId,
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
    }

    // Update only provided fields
    const updatedSettings: PersonalizationSettings = {
      ...existingSettings,
      currentModule: settingsInput.currentModule ?? existingSettings.currentModule,
      currentChapter: settingsInput.currentChapter ?? existingSettings.currentChapter,
      progress: settingsInput.progress ?? existingSettings.progress,
      preferences: {
        ...existingSettings.preferences,
        ...settingsInput.preferences,
      },
      learningHistory: settingsInput.learningHistory ?? existingSettings.learningHistory,
      recommendedContent: settingsInput.recommendedContent ?? existingSettings.recommendedContent,
    };

    mockPersonalizationDb[userId] = updatedSettings;
    return res.status(200).json(updatedSettings);
  } catch (error) {
    console.error('Error updating partial personalization settings:', error);
    return res.status(500).json({ error: 'Failed to update personalization settings' });
  }
}

// Helper function to validate personalization settings input
function validatePersonalizationSettingsInput(input: PersonalizationSettingsInput): boolean {
  // Validate content difficulty
  if (input.preferences?.contentDifficulty &&
      !['adaptive', 'beginner', 'intermediate', 'advanced'].includes(input.preferences.contentDifficulty)) {
    return false;
  }

  // Validate content format
  if (input.preferences?.contentFormat &&
      !['text-heavy', 'visual-heavy', 'balanced'].includes(input.preferences.contentFormat)) {
    return false;
  }

  // Validate notification frequency
  if (input.preferences?.notificationFrequency &&
      !['frequent', 'moderate', 'minimal'].includes(input.preferences.notificationFrequency)) {
    return false;
  }

  // Validate learning pace
  if (input.preferences?.learningPace &&
      !['fast', 'moderate', 'slow'].includes(input.preferences.learningPace)) {
    return false;
  }

  return true;
}

// Additional API endpoints for specific personalization features

// GET /api/personalization/recommendations
export async function getRecommendations(req: NextApiRequest, res: NextApiResponse) {
  const session = await auth.api.getSession({
    headers: req.headers,
  });

  if (!session) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const userId = session.user.id;
  const settings = mockPersonalizationDb[userId];

  if (!settings) {
    return res.status(404).json({ error: 'Personalization settings not found' });
  }

  // Generate recommendations based on user profile and progress
  const recommendations = settings.recommendedContent;

  return res.status(200).json({ recommendations });
}

// POST /api/personalization/progress
export async function updateProgress(req: NextApiRequest, res: NextApiResponse) {
  const session = await auth.api.getSession({
    headers: req.headers,
  });

  if (!session) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const userId = session.user.id;
  const { chapterId, progressPercentage, timeSpent, difficultyRating, feedback } = req.body;

  // Validate input
  if (!chapterId || progressPercentage === undefined) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  let existingSettings = mockPersonalizationDb[userId];
  if (!existingSettings) {
    existingSettings = {
      id: `settings_${Date.now()}`,
      userId,
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
  }

  // Update progress
  const moduleProgress = existingSettings.progress.find(p => p.moduleId === 1); // Simplified for example
  if (moduleProgress) {
    if (!moduleProgress.completedChapters.includes(chapterId)) {
      moduleProgress.completedChapters.push(chapterId);
    }
    moduleProgress.progressPercentage = progressPercentage;
    moduleProgress.timeSpent = (moduleProgress.timeSpent || 0) + (timeSpent || 0);
  } else {
    existingSettings.progress.push({
      moduleId: 1, // Simplified for example
      completedChapters: [chapterId],
      progressPercentage,
      timeSpent: timeSpent || 0,
    });
  }

  // Add to learning history
  existingSettings.learningHistory.push({
    chapterId,
    completedAt: new Date(),
    timeSpent: timeSpent || 0,
    difficultyRating: difficultyRating || 3,
    feedback: feedback || '',
  });

  mockPersonalizationDb[userId] = existingSettings;

  return res.status(200).json({ success: true, progress: existingSettings.progress });
}