import React from 'react';
import Giscus from '@giscus/react';

export const Comment = () => {
  return (
    <div style={{ paddingTop: 50 }}>
      <Giscus
        id="comments"
        // 以下部分参考 Giscus 生成的代码填写
        repo="smiler488/smiler488.github.io" 
        repoId="R_kgDOOA7x0w" 
        category="General"
        categoryId="DIC_kwDOOA7x084CnbRG"
        mapping="pathname"
        strict="0"
        term="Welcome to @giscus/react component!"
        reactionsEnabled="1"
        emitMetadata="0"
        inputPosition="bottom"
        theme="preferred_color_scheme"
        lang="en"
        loading="lazy"
      />
    </div>
  );
};

export default Comment;