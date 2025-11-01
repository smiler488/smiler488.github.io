(function () {
  const LINKS = [
    { href: '/docs/category/tutorial---apps', label: 'Tutorial' },
    { href: '/blog', label: 'Blog' },
    { href: '/app', label: 'App' },
    { href: 'https://github.com/smiler488', label: 'GitHub', external: true },
  ];

  function createLink(link) {
    const a = document.createElement('a');
    a.className = 'mobile-fallback-menu__link';
    a.textContent = link.label;
    a.href = link.href;
    if (link.external) {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    }
    a.addEventListener('click', () => {
      document.body.classList.remove('mobile-fallback-menu--open');
    });
    return a;
  }

  function buildOverlay() {
    if (document.querySelector('.mobile-fallback-menu')) return;

    const overlay = document.createElement('div');
    overlay.className = 'mobile-fallback-menu';

    const panel = document.createElement('div');
    panel.className = 'mobile-fallback-menu__panel';

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'mobile-fallback-menu__close';
    closeBtn.innerHTML = '&times;';
    closeBtn.addEventListener('click', () => {
      document.body.classList.remove('mobile-fallback-menu--open');
    });

    panel.appendChild(closeBtn);

    LINKS.forEach((link) => {
      panel.appendChild(createLink(link));
    });

    overlay.appendChild(panel);
    overlay.addEventListener('click', (event) => {
      if (event.target === overlay) {
        document.body.classList.remove('mobile-fallback-menu--open');
      }
    });

    document.body.appendChild(overlay);
  }

  function handleToggleClick() {
    const toggle = document.querySelector('.navbar__toggle');
    if (!toggle) return;

    toggle.addEventListener(
      'click',
      () => {
        setTimeout(() => {
          if (document.body.classList.contains('navbar-sidebar--show')) {
            document.body.classList.remove('mobile-fallback-menu--open');
          } else {
            document.body.classList.toggle('mobile-fallback-menu--open');
          }
        }, 0);
      },
      { passive: true }
    );
  }

  function init() {
    if (typeof document === 'undefined') return;
    buildOverlay();
    handleToggleClick();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
