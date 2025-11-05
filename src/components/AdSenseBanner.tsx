import {useEffect} from 'react';

declare global {
  interface Window {
    adsbygoogle: unknown[];
  }
}

export default function AdSenseBanner() {
  useEffect(() => {
    try {
      (window.adsbygoogle = window.adsbygoogle || []).push({});
    } catch (err) {
      console.error('Adsense error', err);
    }
  }, []);

  return (
    <ins
      className="adsbygoogle"
      style={{display: 'block'}}
      data-ad-client="ca-pub-7601846275501188"
      data-ad-slot="替换为你的广告单元ID"
      data-ad-format="auto"
      data-full-width-responsive="true"
    />
  );
}