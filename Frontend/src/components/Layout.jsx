import React from 'react';
import { useLocation } from '@docusaurus/router';
import { useBaseUrlUtils } from '@docusaurus/useBaseUrl';
import Head from '@docusaurus/Head';
import Navbar from '@theme/Navbar';
import Footer from '@theme/Footer';
import clsx from 'clsx';

function Layout(props) {
  const {children, className} = props;
  const location = useLocation();
  const {withBaseUrl} = useBaseUrlUtils();
  const isBlogPostPage = location.pathname.includes('/blog/');

  return (
    <>
      <Head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="canonical" href={withBaseUrl(location.pathname)} />
      </Head>
      <Navbar />
      <div className={clsx('main-wrapper', className)}>
        {children}
      </div>
      {!isBlogPostPage && <Footer />}
    </>
  );
}

export default Layout;