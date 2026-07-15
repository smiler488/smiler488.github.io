import React from "react";
import Giscus from "@giscus/react";
import { useColorMode } from "@docusaurus/theme-common";

export const Comment = () => {
  const { colorMode } = useColorMode();

  return (
    <div style={{ paddingTop: 50 }}>
      <Giscus
        id="comments"
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
        theme={colorMode === "dark" ? "dark" : "light"}
        lang="en"
        loading="lazy"
      />
    </div>
  );
};

export default Comment;
