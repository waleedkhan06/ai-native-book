import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1: ROS2 Fundamentals',
      items: [
        'module-1/index',
        'module-1/chapter-1.1',
        'module-1/chapter-1.2',
        'module-1/chapter-1.3',
        'module-1/chapter-1.4',
        'module-1/chapter-1.5'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity Simulation',
      items: [
        'module-2/index',
        'module-2/chapter-2.1',
        'module-2/chapter-2.2',
        'module-2/chapter-2.3'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac Platform',
      items: [
        'module-3/index',
        'module-3/chapter-3.1',
        'module-3/chapter-3.2',
        'module-3/chapter-3.3'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      items: [
        'module-4/index',
        'module-4/chapter-4.1',
        'module-4/chapter-4.2',
        'module-4/chapter-4.3',
        'module-4/chapter-4.4',
        'module-4/chapter-4.5'
      ],
    },
    {
      type: 'category',
      label: 'Hardware Guide',
      items: [
        'hardware/index',
        'hardware/workstation-setup',
        'hardware/jetson-setup',
        'hardware/unitree-setup'
      ],
    },
    {
      type: 'category',
      label: 'Projects',
      items: [
        'projects/index',
        'projects/project-1',
        'projects/project-2',
        'projects/project-3',
        'projects/project-4'
      ],
    },
  ],
};

export default sidebars;
