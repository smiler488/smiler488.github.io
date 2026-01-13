import React from 'react';
import Footer from '@theme-original/Footer';
import styles from './styles.module.css';

// 社交媒体图标配置
const SocialLinks = {
  'CN Community': [
    {
      name: 'Bilibili',
      icon: '/img/Bilibili.png',
      href: 'https://space.bilibili.com/16062789',
      color: '#00a1d6',
    },
    {
      name: 'Douyin',
      icon: '/img/Douyin.png',
      href: 'https://v.douyin.com/-1moIAdEYpg/ 4@5.com :2pm',
      color: '#000000',
    },
    {
      name: 'Weibo',
      icon: '/img/Weibo.png',
      href: 'https://m.weibo.cn/profile/5283742028',
      color: '#e6162d',
    },
    {
      name: 'WeChat',
      icon: '/img/WeChat Offical.png',
      href: 'https://mp.weixin.qq.com/s/JPLLGnM6fwT8XpBdfoXKNA',
      color: '#07c160',
    },
  ],
  'EN Community': [
    {
      name: 'YouTube',
      icon: '/img/YouTube.png',
      href: 'https://www.youtube.com/channel/UCmz7DQ3nEPRxj4rvEQUCvAg',
      color: '#ff0000',
    },
    {
      name: 'TikTok',
      icon: '/img/TikTok.png',
      href: 'https://www.tiktok.com/@smiler488tt',
      color: '#000000',
    },
    {
      name: 'X',
      icon: '/img/X.png',
      href: 'https://x.com/smiler488',
      color: '#000000',
    },
    {
      name: 'Reddit',
      icon: '/img/Reddit.png',
      href: 'https://www.reddit.com/user/smiler488/',
      color: '#ff4500',
    },
  ],
  'More': [
    {
      name: 'LinkedIn',
      icon: '/img/LinkedIn.png',
      href: 'https://www.linkedin.com/in/liangchao-deng-7b420b269/',
      color: '#0077b5',
    },
    {
      name: 'HuggingFace',
      icon: '/img/HuggingFace.png',
      href: 'https://huggingface.co/smiler488',
      color: '#ff9d00',
    },
    {
      name: 'Bluesky',
      icon: '/img/Bluesky.png',
      href: 'https://bsky.app/profile/smiler488.bsky.social',
      color: '#1285fe',
    },
    {
      name: 'GitHub',
      icon: '/img/Github.png',
      href: 'https://github.com/smiler488',
      color: '#24292e',
    },
  ],
};

// 社交媒体图标组件
function SocialIcon({ name, icon, href, color }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.socialIcon}
      title={name}
      style={{ '--hover-color': color }}
    >
      <img src={icon} alt={name} />
    </a>
  );
}

// 自定义页脚组件
export default function FooterWrapper(props) {
  return (
    <>
      {/* 社交媒体区域 */}
      <div className={styles.socialFooter}>
        <div className="container">
          <div className={styles.socialGroups}>
            {Object.entries(SocialLinks).map(([groupName, links]) => (
              <div key={groupName} className={styles.socialGroup}>
                <h3 className={styles.groupTitle}>{groupName}</h3>
                <div className={styles.socialIcons}>
                  {links.map((social) => (
                    <SocialIcon key={social.name} {...social} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 原始页脚 */}
      <Footer {...props} />
    </>
  );
}
