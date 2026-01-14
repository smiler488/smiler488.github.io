import React from 'react';

export default function Root({children}) {
  // 注意：Algolia DocSearch 已移除，现在使用 @easyops-cn/docusaurus-search-local 插件
  // 该插件会自动处理搜索功能，无需手动初始化

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