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

  return <>{children}</>;
}