import React, { ReactNode } from 'react';
import clsx from 'clsx';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import styles from './CodeExample.module.css';

type CodeExampleProps = {
  title?: string;
  description?: string;
  children: ReactNode;
  language?: string;
  fileName?: string;
  isRunnable?: boolean;
  className?: string;
};

const CodeExample = ({
  title,
  description,
  children,
  language = 'python',
  fileName,
  isRunnable = false,
  className,
}: CodeExampleProps): JSX.Element => {
  const getLanguageDisplayName = (lang: string) => {
    const langMap: Record<string, string> = {
      python: 'Python',
      xml: 'XML/URDF',
      cpp: 'C++',
      bash: 'Bash',
      yaml: 'YAML',
      launch: 'Launch File',
    };
    return langMap[lang] || lang.toUpperCase();
  };

  return (
    <div className={clsx(styles.codeExampleContainer, className)}>
      {(title || fileName) && (
        <div className={styles.codeHeader}>
          {title && <h3 className={styles.codeTitle}>{title}</h3>}
          {fileName && <span className={styles.fileName}>{fileName}</span>}
          {isRunnable && (
            <span className={clsx(styles.runnableIndicator, styles.runnable)}>
              Runnable
            </span>
          )}
        </div>
      )}
      {description && <p className={styles.codeDescription}>{description}</p>}
      <div className={styles.codeBlock}>
        <pre className={clsx('language-' + language, styles.codeBlockPre)}>
          <code className={clsx('language-' + language, styles.codeBlockCode)}>
            {children}
          </code>
        </pre>
      </div>
    </div>
  );
};

// For multiple language examples
type MultiLanguageCodeExampleProps = {
  title?: string;
  description?: string;
  examples: {
    language: string;
    code: string;
    fileName?: string;
    isRunnable?: boolean;
  }[];
  className?: string;
};

const MultiLanguageCodeExample = ({
  title,
  description,
  examples,
  className,
}: MultiLanguageCodeExampleProps): JSX.Element => {
  if (examples.length === 1) {
    const example = examples[0];
    return (
      <CodeExample
        title={title}
        description={description}
        language={example.language}
        fileName={example.fileName}
        isRunnable={example.isRunnable}
        className={className}
      >
        {example.code}
      </CodeExample>
    );
  }

  return (
    <div className={clsx(styles.multiCodeExampleContainer, className)}>
      {title && <h3 className={styles.multiCodeTitle}>{title}</h3>}
      {description && <p className={styles.codeDescription}>{description}</p>}
      <Tabs groupId="code-examples">
        {examples.map((example, index) => (
          <TabItem
            key={index}
            value={example.language}
            label={example.fileName || example.language.toUpperCase()}
          >
            <div className={styles.codeBlock}>
              <pre className={clsx('language-' + example.language, styles.codeBlockPre)}>
                <code className={clsx('language-' + example.language, styles.codeBlockCode)}>
                  {example.code}
                </code>
              </pre>
            </div>
          </TabItem>
        ))}
      </Tabs>
    </div>
  );
};

export { CodeExample, MultiLanguageCodeExample };