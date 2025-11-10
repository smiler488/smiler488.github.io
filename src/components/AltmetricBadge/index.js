import React, { useEffect } from 'react';

const AltmetricBadge = ({
  doi,
  badgeType = 'donut',
  hideNoMentions = false,
  className = '',
}) => {
  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const initAltmetric = () => {
      if (typeof window._altmetric_embed_init === 'function') {
        window._altmetric_embed_init();
      }
    };

    const scriptId = 'altmetric-embed-script';
    const existingScript = document.getElementById(scriptId);

    if (existingScript) {
      initAltmetric();
      return;
    }

    const script = document.createElement('script');
    script.id = scriptId;
    script.src = 'https://badges.altmetric.com/embed.js';
    script.async = true;
    script.onload = initAltmetric;
    document.body.appendChild(script);
  }, [doi, badgeType, hideNoMentions]);

  return (
    <div
      className={`altmetric-embed ${className}`.trim()}
      data-badge-type={badgeType}
      data-doi={doi}
      data-hide-no-mentions={hideNoMentions}
    />
  );
};

export default AltmetricBadge;
