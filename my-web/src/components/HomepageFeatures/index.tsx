import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';
import {FaBook, FaRobot, FaCode, FaTools} from 'react-icons/fa';

type FeatureItem = {
  title: string;
  description: ReactNode;
  icon: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: '🤖 AI-Powered Learning',
    icon: <FaRobot size={40} color="#8B5CF6" />,
    description: (
      <>
        Interactive AI tutors and personalized learning paths to help you master robotics concepts at your own pace.
      </>
    ),
  },
  {
    title: '📚 Comprehensive Curriculum',
    icon: <FaBook size={40} color="#8B5CF6" />,
    description: (
      <>
        Complete coverage from fundamentals to advanced topics in physical AI and humanoid robotics.
      </>
    ),
  },
  {
    title: '🔧 Hands-on Projects',
    icon: <FaCode size={40} color="#8B5CF6" />,
    description: (
      <>
        Practical exercises and real-world projects to apply theoretical knowledge in simulation and hardware.
      </>
    ),
  },
  {
    title: '🛠️ Advanced Tools',
    icon: <FaTools size={40} color="#8B5CF6" />,
    description: (
      <>
        Cutting-edge tools and frameworks to build and deploy your AI and robotics projects.
      </>
    ),
  },
];

function Feature({title, description, icon}: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center padding-horiz--md">
        <div className="feature-card">
          <div className="feature-icon">
            {icon}
          </div>
          <Heading as="h3">{title}</Heading>
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
