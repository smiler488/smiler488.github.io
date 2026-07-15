export default {
  extends: ["stylelint-config-standard"],
  rules: {
    // The existing site intentionally mixes Docusaurus/BEM globals, CSS Modules,
    // legacy rgba syntax, and prefixed backdrop-filter for Safari. Keep linting
    // focused on invalid CSS instead of forcing a repository-wide style rewrite.
    "alpha-value-notation": null,
    "color-function-alias-notation": null,
    "color-function-notation": null,
    "color-hex-length": null,
    "comment-empty-line-before": null,
    "custom-property-empty-line-before": null,
    "declaration-block-no-duplicate-custom-properties": null,
    "declaration-block-no-redundant-longhand-properties": null,
    "declaration-block-single-line-max-declarations": null,
    "font-family-name-quotes": null,
    "keyframes-name-pattern": null,
    "media-feature-range-notation": null,
    "no-descending-specificity": null,
    "no-duplicate-selectors": null,
    "property-no-vendor-prefix": null,
    "rule-empty-line-before": null,
    "selector-class-pattern": null,
    "selector-id-pattern": null,
    "selector-not-notation": null,
    "selector-pseudo-class-no-unknown": [
      true,
      { ignorePseudoClasses: ["global"] },
    ],
    "shorthand-property-no-redundant-values": null,
    "value-keyword-case": null,
  },
};
