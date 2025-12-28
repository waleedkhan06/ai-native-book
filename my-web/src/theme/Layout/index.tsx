import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import Chatbot from '@site/src/components/Chatbot';

export default function LayoutWrapper(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <Chatbot />
    </>
  );
}
