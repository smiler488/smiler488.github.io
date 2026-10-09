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

    // Design-system guardrails (design/DESIGN_SPEC.md §5.6, §5.8).
    // Colours come from src/css/tokens.css; no frosted glass, no all-caps labels,
    // no decorative radial glows.
    "color-no-hex": true,
    "declaration-property-value-disallowed-list": [
      {
        "text-transform": ["uppercase"],
        background: ["/radial-gradient/"],
        "background-image": ["/radial-gradient/"],
        "backdrop-filter": ["/blur/"],
        "-webkit-backdrop-filter": ["/blur/"],
      },
      {
        message:
          "Sentence case, no decorative glows, no frosted glass (DESIGN_SPEC §5.6). Floating layers opt out with a commented stylelint-disable.",
      },
    ],
    "declaration-property-value-allowed-list": [
      { "font-weight": ["400", "500", "600", "normal", "inherit"] },
      { message: "Use font weights 400 / 500 / 600 only (DESIGN_SPEC §5.2)." },
    ],
  },
  overrides: [
    {
      // The token file is the one place raw colour values live.
      files: ["src/css/tokens.css"],
      rules: { "color-no-hex": null },
    },
    {
      // Tool-local canvas and chart colours; converged in P3.
      files: ["src/pages/app/**/*.css"],
      rules: { "color-no-hex": null },
    },
    {
      // Standalone visual-experiment page; exempt from the quiet-UI rules.
      files: ["src/components/HologramParticles/**/*.css"],
      rules: {
        "color-no-hex": null,
        "declaration-property-value-disallowed-list": null,
        "declaration-property-value-allowed-list": null,
      },
    },
  ],
};
