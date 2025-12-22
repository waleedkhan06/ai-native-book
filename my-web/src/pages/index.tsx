import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import {FaBook, FaRobot, FaCode, FaTools} from 'react-icons/fa';
import ParticleBackground from '@site/src/components/ParticleBackground';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title" style={{ color: '#8B5CF6' }}>
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle" style={{ color: 'white' }}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            style={{ backgroundColor: '#A78BFA', borderColor: '#A78BFA' }}
            to="/docs/intro">
            <FaBook className={styles.buttonIcon} /> Explore Textbook
          </Link>
          <span className={styles.spacer}></span>
          <Link
            className="button button--secondary button--lg"
            style={{ backgroundColor: '#7C3AED', borderColor: '#7C3AED' }}
            to="/docs/tutorial">
            <FaRobot className={styles.buttonIcon} /> AI Book Tutorial
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />"
      style={{ backgroundColor: '#111827' }}>
      <div style={{ position: 'relative', minHeight: '100vh', backgroundColor: '#111827', color: 'white' }}>
        <ParticleBackground />
        <HomepageHeader />
        <main style={{ position: 'relative', zIndex: 1 }}>
          <HomepageFeatures />
        </main>
      </div>
    </Layout>
  );
}
