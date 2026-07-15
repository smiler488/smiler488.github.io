module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  extends: [
    "plugin:@docusaurus/recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "prettier",
  ],
  parserOptions: {
    ecmaVersion: "latest",
    ecmaFeatures: { jsx: true },
    sourceType: "module",
  },
  settings: {
    react: { version: "detect" },
  },
  ignorePatterns: [
    ".docusaurus/",
    "build/",
    "node_modules/",
    "static/lib/",
    "static/resources/",
    "instantsearch-app/",
  ],
  rules: {
    "react/prop-types": "off",
    "react/react-in-jsx-scope": "off",
    "react-hooks/set-state-in-effect": "warn",
  },
};
