import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function Root({children}) {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === 'zh-Hans';

  // 导航栏滚动动态模糊效果
  React.useEffect(() => {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    let lastScrollY = window.scrollY;
    let ticking = false;

    const updateNavbar = () => {
      const scrollY = window.scrollY;

      if (scrollY > 20) {
        navbar.classList.add('navbar--scrolled');
      } else {
        navbar.classList.remove('navbar--scrolled');
      }

      lastScrollY = scrollY;
      ticking = false;
    };

    const onScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(updateNavbar);
        ticking = true;
      }
    };

    // 初始设置
    updateNavbar();

    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const WECHAT_URL = 'https://work.weixin.qq.com/kfid/kfc63941027aeefc636';

  return (
    <>
      {children}
      <div className="wechat-float-container">
        {/* Compact QR card revealed on hover / focus */}
        <div className="wechat-float-qr" role="tooltip">
          <img
            src="/img/wechat-qr.png"
            alt={isChinese ? '微信客服二维码' : 'WeChat QR Code'}
            width="112"
            height="112"
            loading="lazy"
          />
          <p className="wechat-float-qr-hint">
            {isChinese ? '微信扫码咨询' : 'Scan to chat on WeChat'}
          </p>
          <Link
            className="wechat-float-qr-link"
            to={WECHAT_URL}
            target="_blank"
            rel="noopener noreferrer"
          >
            {isChinese ? '或点击直接联系 →' : 'or click to contact →'}
          </Link>
        </div>

        {/* Floating circular button */}
        <Link
          className="wechat-float-btn"
          to={WECHAT_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={isChinese ? '微信客服' : 'WeChat Support'}
        >
          <svg
            className="wechat-float-btn-icon"
            width="22"
            height="22"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M8.28 2.05C4.24 2.05 1 4.79 1 8.24c0 1.95 1.05 3.73 2.72 4.9L3.08 15.6l2.94-1.46c.72.2 1.48.31 2.26.31 4.04 0 7.28-2.74 7.28-6.2S12.32 2.05 8.28 2.05zm-3.03 5.4c-.45 0-.82-.36-.82-.82s.36-.82.82-.82.82.37.82.82-.36.82-.82.82zm5.72 0c-.45 0-.82-.36-.82-.82s.36-.82.82-.82.82.37.82.82-.36.82-.82.82zm9.18 4.29c3.34 0 6.02-2.27 6.02-5.13S23.49 1.5 20.15 1.5c-3.34 0-6.02 2.27-6.02 5.13 0 2.87 2.68 5.13 6.02 5.13zm-2.45-4.47c-.37 0-.68-.3-.68-.68s.3-.68.68-.68.68.3.68.68-.3.68-.68.68zm4.77 0c-.37 0-.68-.3-.68-.68s.3-.68.68-.68.68.3.68.68-.3.68-.68.68z" />
          </svg>
        </Link>
      </div>
    </>
  );
}