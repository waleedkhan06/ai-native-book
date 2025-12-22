/**
 * Learning Path Selector Component
 * Allows users to select and switch between different learning paths
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from 'docusaurus-plugin-better-auth/client';
import { PersonalizationService } from '../services/personalization-service';
import { UserProfile } from '../types/user-profile';

interface LearningPathOption {
  id: string;
  title: string;
  description: string;
  icon: string;
  estimatedDuration: string;
  prerequisites: string[];
  targetAudience: string[];
}

interface LearningPathSelectorProps {
  onPathChange?: (path: string) => void;
}

const LearningPathSelector: React.FC<LearningPathSelectorProps> = ({ onPathChange }) => {
  const { session, update } = useAuth();
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [availablePaths, setAvailablePaths] = useState<LearningPathOption[]>([]);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);

  useEffect(() => {
    if (session?.user) {
      const profile: UserProfile = {
        id: session.user.id,
        email: session.user.email || '',
        name: session.user.name || '',
        createdAt: new Date(),
        updatedAt: new Date(),
        expertiseLevel: session.user.expertiseLevel as 'beginner' | 'intermediate' | 'advanced' || 'beginner',
        backgroundQuestions: {
          technicalBackground: session.user.technicalBackground || 'none',
          roboticsExperience: session.user.roboticsExperience || 'no-experience',
          primaryGoal: session.user.primaryGoal || 'learn-basics',
          hardwareAccess: session.user.hardwareAccess || 'no-hardware',
          timeCommitment: session.user.timeCommitment || 'less-than-2-hours',
        },
        preferredLanguage: session.user.preferredLanguage as 'en' | 'ur' || 'en',
        learningPath: session.user.learningPath || 'general',
      };
      setUserProfile(profile);
      setSelectedPath(profile.learningPath);

      // Generate available learning paths based on user profile
      const paths = generateLearningPaths(profile);
      setAvailablePaths(paths);
    }
  }, [session]);

  const generateLearningPaths = (profile: UserProfile): LearningPathOption[] => {
    return [
      {
        id: 'foundational',
        title: 'Foundational Robotics',
        description: 'Start with the basics of ROS2, robot architecture, and fundamental concepts',
        icon: '📚',
        estimatedDuration: '4-6 weeks',
        prerequisites: ['Basic programming knowledge'],
        targetAudience: ['Complete beginners', 'Those new to robotics'],
      },
      {
        id: 'project-focused-simulation',
        title: 'Project-Based (Simulation)',
        description: 'Learn robotics by building projects in simulation environments like Gazebo',
        icon: '💻',
        estimatedDuration: '6-8 weeks',
        prerequisites: ['Basic programming', 'Familiarity with Linux'],
        targetAudience: ['Learners without hardware access', 'Project-oriented learners'],
      },
      {
        id: 'project-focused-hardware',
        title: 'Project-Based (Hardware)',
        description: 'Apply concepts to real hardware platforms like Unitree robots',
        icon: '🤖',
        estimatedDuration: '8-10 weeks',
        prerequisites: ['Access to robotics hardware', 'Intermediate programming'],
        targetAudience: ['Learners with hardware access', 'Hands-on learners'],
      },
      {
        id: 'career-transition',
        title: 'Career Transition',
        description: 'Comprehensive path for moving into robotics industry',
        icon: '💼',
        estimatedDuration: '10-12 weeks',
        prerequisites: ['Technical background', 'Dedicated time commitment'],
        targetAudience: ['Career changers', 'Industry professionals'],
      },
      {
        id: 'academic',
        title: 'Academic Deep Dive',
        description: 'Theory-heavy approach with mathematical foundations',
        icon: '🎓',
        estimatedDuration: '12-16 weeks',
        prerequisites: ['Mathematical background', 'Research interest'],
        targetAudience: ['Students', 'Researchers', 'Academic learners'],
      },
    ];
  };

  const handlePathSelect = async (pathId: string) => {
    setSelectedPath(pathId);

    if (onPathChange) {
      onPathChange(pathId);
    }

    // Update user's learning path in their profile
    if (session?.user && update) {
      try {
        await update({
          learningPath: pathId,
        });
      } catch (error) {
        console.error('Error updating learning path:', error);
      }
    }
  };

  return (
    <div className="learning-path-selector p-6 bg-gray-50 rounded-lg">
      <h3 className="text-xl font-bold mb-4">Choose Your Learning Path</h3>
      <p className="mb-6 text-gray-600">
        Based on your background and goals, we've customized these learning paths for you.
        You can change this anytime.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {availablePaths.map((path) => (
          <div
            key={path.id}
            className={`border rounded-lg p-4 cursor-pointer transition-all ${
              selectedPath === path.id
                ? 'border-blue-500 bg-blue-50 shadow-md'
                : 'border-gray-200 hover:border-blue-300 hover:bg-gray-100'
            }`}
            onClick={() => handlePathSelect(path.id)}
          >
            <div className="flex items-start">
              <span className="text-2xl mr-3">{path.icon}</span>
              <div>
                <h4 className="font-bold text-lg">{path.title}</h4>
                <p className="text-gray-600 text-sm mt-1">{path.description}</p>

                <div className="mt-3 text-xs">
                  <div className="flex items-center mt-2">
                    <span className="font-medium">Duration:</span>
                    <span className="ml-2 text-gray-600">{path.estimatedDuration}</span>
                  </div>

                  <div className="flex items-center mt-1">
                    <span className="font-medium">For:</span>
                    <span className="ml-2 text-gray-600">{path.targetAudience.join(', ')}</span>
                  </div>
                </div>

                {selectedPath === path.id && (
                  <div className="mt-3">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      Selected
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {selectedPath && (
        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <h4 className="font-bold text-blue-800">Your Selected Path: {availablePaths.find(p => p.id === selectedPath)?.title}</h4>
          <p className="text-blue-700 mt-2">
            Your textbook content will be personalized based on this learning path.
            You can switch paths at any time.
          </p>
        </div>
      )}
    </div>
  );
};

export default LearningPathSelector;