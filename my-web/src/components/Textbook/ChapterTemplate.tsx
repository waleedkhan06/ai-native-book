import React, { ReactNode } from 'react';
import clsx from 'clsx';
import styles from './ChapterTemplate.module.css';

type ChapterTemplateProps = {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
};

const ChapterTemplate = ({
  title,
  description,
  children,
  className,
}: ChapterTemplateProps): JSX.Element => {
  return (
    <div className={clsx('container', styles.chapterContainer, className)}>
      <header className={styles.chapterHeader}>
        <h1 className={styles.chapterTitle}>{title}</h1>
        {description && <p className={styles.chapterDescription}>{description}</p>}
      </header>
      <main className={styles.chapterContent}>{children}</main>
    </div>
  );
};

export default ChapterTemplate;