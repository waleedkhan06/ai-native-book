import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'The Complete Guide to Embodied Intelligence',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },
  // Set the production url of your site here
  url: 'https://waleedkhan06.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/ai-native-book/',

  // Add trailingSlash config to avoid GitHub Pages redirect issues
  trailingSlash: false,

  onBrokenLinks: 'warn', // Change from default 'throw' to 'warn' to allow deployment
  onBrokenMarkdownLinks: 'warn',

  //  GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'waleedkhan06', // Usually your GitHub org/user name.
  projectName: 'ai-native-book', // Usually your repo name.

  // onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/haclathon/ai-native-book/tree/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        gtag: {
          trackingID: 'G-XXXXXXXXXX',
          anonymizeIP: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: true,  // Remove light/dark mode toggle
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Logo',
        src: 'img/logo.svg',
      },
      style: 'dark',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/haclathon/ai-native-book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
          items: [
            {
              label: 'Module 1: ROS2 Fundamentals',
              to: '/docs/module-1',
            },
            {
              label: 'Module 2: Gazebo & Unity Simulation',
              to: '/docs/module-2',
            },
            {
              label: 'Module 3: NVIDIA Isaac Platform',
              to: '/docs/module-3',
            },
            {
              label: 'Module 4: Vision-Language-Action',
              to: '/docs/module-4',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Projects',
              to: '/docs/projects',
            },
            {
              label: 'Hardware Guide',
              to: '/docs/hardware',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/haclathon/ai-native-book',
            },
            {
              label: 'Discord',
              href: 'https://discord.gg/example',
            },
            {
              label: 'Twitter',
              href: 'https://twitter.com/example',
            },
            {
              label: 'Facebook',
              href: 'https://facebook.com/example',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;