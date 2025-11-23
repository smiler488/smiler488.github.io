import React from 'react';
import docsearch from '@docsearch/js';
import '@docsearch/css';

export default function Root({children}) {
  React.useEffect(() => {
    const navRight = document.querySelector('.navbar__items--right');
    const host = navRight || document.querySelector('.navbar__items--left') || document.querySelector('.navbar');
    if (!host) return;
    let container = document.getElementById('docsearch');
    if (!container) {
      container = document.createElement('div');
      container.id = 'docsearch';
      container.className = 'docsearch-input';
      host.insertBefore(container, host.firstChild);
    }
    docsearch({
      container: '#docsearch',
      appId: 'HITKU3S49E',
      indexName: 'smiler488.github.io',
      apiKey: '3b2ba7e1f5fbb72755b79c9a7c616457',
      askAi: 'YOUR_ALGOLIA_ASSISTANT_ID',
    });
    return () => {
      const el = document.getElementById('docsearch');
      if (el && el.parentElement) el.parentElement.removeChild(el);
    };
  }, []);
  return <>{children}</>;
}