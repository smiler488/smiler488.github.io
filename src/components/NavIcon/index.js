/**
 * Navbar icons. Each has an outline form (resting) and a filled form (hover,
 * focus and the current page); the navbar crossfades between the two.
 */
import React from "react";

const LINE = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

const ICONS = {
  // Research: a sprout, for plant science
  research: {
    outline: (
      <g {...LINE}>
        <path d="M12 21v-9" />
        <path d="M12 12c0-4.4 3.1-7.5 8-7.5 0 4.4-3.1 7.5-8 7.5Z" />
        <path d="M12 14.5c0-3.5-2.5-6-7.5-6 0 3.5 2.5 6 7.5 6Z" />
      </g>
    ),
    solid: (
      <g {...LINE}>
        <path d="M12 21v-9" />
        <path
          d="M12 12c0-4.4 3.1-7.5 8-7.5 0 4.4-3.1 7.5-8 7.5Z"
          fill="currentColor"
        />
        <path
          d="M12 14.5c0-3.5-2.5-6-7.5-6 0 3.5 2.5 6 7.5 6Z"
          fill="currentColor"
        />
      </g>
    ),
  },
  // Lab: a flask; filled, it holds liquid
  lab: {
    outline: (
      <g {...LINE}>
        <path d="M9 3h6" />
        <path d="M10 3v6.2l-5.2 9a1.9 1.9 0 0 0 1.6 2.8h11.2a1.9 1.9 0 0 0 1.6-2.8l-5.2-9V3" />
      </g>
    ),
    solid: (
      <g {...LINE}>
        <path d="M9 3h6" />
        <path d="M10 3v6.2l-5.2 9a1.9 1.9 0 0 0 1.6 2.8h11.2a1.9 1.9 0 0 0 1.6-2.8l-5.2-9V3" />
        <path
          d="M7.3 14h9.4l2.5 4.2a1.9 1.9 0 0 1-1.6 2.8H6.4a1.9 1.9 0 0 1-1.6-2.8Z"
          fill="currentColor"
        />
      </g>
    ),
  },
  // Notes: a page with text lines
  notes: {
    outline: (
      <g {...LINE}>
        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
        <path d="M14 3v5h5" />
        <path d="M9 13h6M9 17h4" />
      </g>
    ),
    solid: (
      <g {...LINE}>
        <path
          d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"
          fill="currentColor"
        />
        {/* Cut-out lines in the surface colour (var() needs the style prop) */}
        <path d="M9 13h6M9 17h4" style={{ stroke: "var(--ds-card)" }} />
      </g>
    ),
  },
  // About: a person
  about: {
    outline: (
      <g {...LINE}>
        <circle cx="12" cy="8" r="4" />
        <path d="M4.5 21a7.5 7.5 0 0 1 15 0" />
      </g>
    ),
    solid: (
      <g {...LINE}>
        <circle cx="12" cy="8" r="4" fill="currentColor" />
        <path d="M4.5 21a7.5 7.5 0 0 1 15 0Z" fill="currentColor" />
      </g>
    ),
  },
  github: {
    outline: (
      <g {...LINE}>
        <path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21" />
      </g>
    ),
    solid: (
      <path
        fill="currentColor"
        transform="translate(0.75 0.75) scale(1.40625)"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
      />
    ),
  },
};

// Navbar labels (both locales) to icon names
const LABELS = {
  Research: "research",
  研究: "research",
  Lab: "lab",
  实验室: "lab",
  Notes: "notes",
  笔记: "notes",
  About: "about",
  关于: "about",
  GitHub: "github",
};

export function navIconName(label) {
  return LABELS[label] ?? null;
}

/**
 * Renders both forms stacked; the caller's CSS shows one of them.
 * `outlineClassName` / `solidClassName` let each navbar style the swap.
 */
export default function NavIcon({ name, outlineClassName, solidClassName }) {
  const icon = ICONS[name];
  if (!icon) return null;
  return (
    <>
      <svg
        className={outlineClassName}
        viewBox="0 0 24 24"
        width="20"
        height="20"
        aria-hidden="true"
        focusable="false"
      >
        {icon.outline}
      </svg>
      <svg
        className={solidClassName}
        viewBox="0 0 24 24"
        width="20"
        height="20"
        aria-hidden="true"
        focusable="false"
      >
        {icon.solid}
      </svg>
    </>
  );
}
