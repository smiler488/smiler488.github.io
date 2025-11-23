import React from 'react';

export default function VisitorMap({ mountInToc = true }) {
  const inlineRef = React.useRef(null);
  React.useEffect(() => {
    const selectors = [
      '.theme-doc-toc-desktop',
      '.table-of-contents',
      'aside[class*="toc"]',
    ];
    const toc = mountInToc ? selectors.map(s => document.querySelector(s)).find(Boolean) : null;
    let targetEl = null;
    if (toc && toc.parentElement) {
      const container = document.createElement('div');
      container.className = 'tocVisitorsContainer';
      toc.parentElement.appendChild(container);
      targetEl = document.createElement('div');
      targetEl.className = 'tocVisitorMap';
      container.appendChild(targetEl);
    } else {
      targetEl = inlineRef.current;
    }
    if (!targetEl) return;
    while (targetEl.firstChild) targetEl.removeChild(targetEl.firstChild);
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.id = 'mapmyvisitors';
    script.src = 'https://mapmyvisitors.com/map.js?d=ccC5JZBvNNpRHfn94y3CRXzvvcSb99CMKXuy-7wzczI&cl=ffffff&w=a';
    script.async = true;
    targetEl.appendChild(script);
    return () => {
      if (targetEl) {
        while (targetEl.firstChild) targetEl.removeChild(targetEl.firstChild);
        const container = targetEl.parentElement;
        const parent = container && container.parentElement;
        if (toc && container && parent === toc.parentElement) parent.removeChild(container);
      }
    };
  }, [mountInToc]);

  return <div className="visitorMapWidget" ref={inlineRef} style={{ display: mountInToc ? 'none' : undefined }} />;
}