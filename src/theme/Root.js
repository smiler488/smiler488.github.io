import React from 'react';
import docsearch from '@docsearch/js';
import '@docsearch/css';

export default function Root({children}) {
  // DocSearch初始化 - 使用 ref 确保只初始化一次
  const docSearchInitialized = React.useRef(false);

  React.useEffect(() => {
    // 避免重复初始化
    if (docSearchInitialized.current) return;

    // 延迟初始化，确保导航栏已渲染
    const initDocSearch = () => {
      const navRight = document.querySelector('.navbar__items--right');
      const host = navRight || document.querySelector('.navbar__items--left') || document.querySelector('.navbar');

      if (!host) {
        // 导航栏还未渲染，重试
        setTimeout(initDocSearch, 100);
        return;
      }

      // 检查是否已存在容器
      let container = document.getElementById('docsearch');
      if (!container) {
        container = document.createElement('div');
        container.id = 'docsearch';
        container.className = 'docsearch-input';
        host.insertBefore(container, host.firstChild);
      }

      try {
        docsearch({
          container: '#docsearch',
          appId: 'HITKU3S49E',
          indexName: 'smiler488github',
          apiKey: '3b2ba7e1f5fbb72755b79c9a7c616457',
          askAi: 'AHa325oRsRLD',
        });
        docSearchInitialized.current = true;
      } catch (error) {
        console.error('DocSearch initialization failed:', error);
      }
    };

    // 在下一帧初始化，确保 DOM 已准备好
    requestAnimationFrame(() => {
      setTimeout(initDocSearch, 0);
    });

    // 不再清理，让搜索框保持稳定
    // 只在组件卸载时清理
    return () => {
      // 页面卸载时才清理
      docSearchInitialized.current = false;
    };
  }, []);

  // 导航栏滚动动态模糊效果
  React.useEffect(() => {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    let lastScrollY = window.scrollY;
    let ticking = false;

    const updateNavbar = () => {
      const scrollY = window.scrollY;

      // 根据滚动距离动态调整模糊和透明度
      if (scrollY > 50) {
        // 滚动后增强玻璃效果
        navbar.style.backdropFilter = 'blur(20px) saturate(180%)';
        navbar.style.webkitBackdropFilter = 'blur(20px) saturate(180%)';
        navbar.style.boxShadow = '0 2px 12px rgba(0, 0, 0, 0.08)';
      } else {
        // 顶部时的轻微模糊
        const blurAmount = Math.min(20, scrollY / 2.5);
        navbar.style.backdropFilter = `blur(${blurAmount}px)`;
        navbar.style.webkitBackdropFilter = `blur(${blurAmount}px)`;
        navbar.style.boxShadow = `0 1px ${scrollY / 10}px rgba(0, 0, 0, 0.05)`;
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

  return <>{children}</>;
}