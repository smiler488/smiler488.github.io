import React from "react";
import clsx from "clsx";
import {
  ErrorCauseBoundary,
  ThemeClassNames,
  useColorMode,
  useThemeConfig,
} from "@docusaurus/theme-common";
import {
  splitNavbarItems,
  useNavbarMobileSidebar,
} from "@docusaurus/theme-common/internal";
import NavbarItem from "@theme/NavbarItem";
import NavbarColorModeToggle from "@theme/Navbar/ColorModeToggle";
import NavbarMobileSidebarToggle from "@theme/Navbar/MobileSidebar/Toggle";
import NavbarLogo from "@theme/Navbar/Logo";
import NavbarSearch from "@theme/Navbar/Search";
import SearchBar from "@theme/SearchBar";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./styles.module.css";

// Tooltip for the theme button: names the current choice (the button opens
// a system / light / dark menu, DESIGN_SPEC D-020).
const THEME_LABEL = {
  en: { system: "Theme: system", light: "Theme: light", dark: "Theme: dark" },
  zh: { system: "主题：跟随系统", light: "主题：浅色", dark: "主题：深色" },
};

function useNavbarItems() {
  return useThemeConfig().navbar.items;
}

function NavbarItems({ items }) {
  return (
    <>
      {items.map((item, index) => (
        <ErrorCauseBoundary
          key={`${item.type || "link"}-${item.label || item.to || index}`}
          onError={(error) =>
            new Error(
              `A theme navbar item failed to render. Please check: ${JSON.stringify(
                item
              )}`,
              { cause: error }
            )
          }
        >
          <NavbarItem {...item} />
        </ErrorCauseBoundary>
      ))}
    </>
  );
}

/**
 * Label behaviour for the icon buttons (muse-design v1.1): Escape hides a
 * visible label until the pointer or focus moves again; a touch keeps the
 * tapped control's label (and filled icon) until the next tap elsewhere.
 * The click itself is untouched, so one tap still acts.
 */
function useIconLabels(ref) {
  React.useEffect(() => {
    const bar = ref.current;
    if (!bar) return undefined;
    const clearTouch = () =>
      bar
        .querySelectorAll("[data-touch-label]")
        .forEach((el) => delete el.dataset.touchLabel);
    const onKey = (event) => {
      if (event.key === "Escape") {
        bar.dataset.labelsDismissed = "true";
        clearTouch();
      }
    };
    const restore = () => delete bar.dataset.labelsDismissed;
    const onPointerDown = (event) => {
      clearTouch();
      if (event.pointerType !== "touch") return;
      const control = event.target.closest?.("a, button");
      if (control && bar.contains(control)) control.dataset.touchLabel = "true";
    };
    const onPointerMove = (event) => {
      if (event.pointerType === "mouse") restore();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointerDown);
    bar.addEventListener("pointermove", onPointerMove);
    bar.addEventListener("focusin", restore);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointerDown);
      bar.removeEventListener("pointermove", onPointerMove);
      bar.removeEventListener("focusin", restore);
    };
  }, [ref]);
}

function NavbarContentLayout({ left, right }) {
  const barRef = React.useRef(null);
  useIconLabels(barRef);
  return (
    <div
      ref={barRef}
      className={clsx("navbar__inner", styles.navbarSurface)}
    >
      <div
        className={clsx(
          ThemeClassNames.layout.navbar.containerLeft,
          "navbar__items",
          styles.leftRail
        )}
      >
        {left}
      </div>
      <div
        className={clsx(
          ThemeClassNames.layout.navbar.containerRight,
          "navbar__items navbar__items--right",
          styles.rightRail
        )}
      >
        {right}
      </div>
    </div>
  );
}

export default function NavbarContent() {
  const { i18n } = useDocusaurusContext();
  const mobileSidebar = useNavbarMobileSidebar();
  const items = useNavbarItems();
  const [leftItems, rightItems] = splitNavbarItems(items);
  const hasConfiguredSearch = items.some((item) => item.type === "search");
  const isChinese = i18n.currentLocale === "zh-Hans";
  const { colorModeChoice } = useColorMode();
  const themeLabel = (isChinese ? THEME_LABEL.zh : THEME_LABEL.en)[
    colorModeChoice ?? "system"
  ];

  return (
    <NavbarContentLayout
      left={
        <>
          {!mobileSidebar.disabled && <NavbarMobileSidebarToggle />}
          <div className={styles.brand}>
            <NavbarLogo />
          </div>
          <nav
            className={styles.primaryNav}
            aria-label={isChinese ? "主要页面" : "Primary pages"}
          >
            <NavbarItems items={leftItems} />
          </nav>
        </>
      }
      right={
        <div className={styles.utilityNav}>
          {!hasConfiguredSearch && (
            <div
              className={clsx(styles.searchSlot, styles.iconButton)}
              data-tooltip={isChinese ? "搜索" : "Search"}
            >
              <NavbarSearch>
                <SearchBar />
              </NavbarSearch>
            </div>
          )}
          <div className={styles.rightItems}>
            <NavbarItems items={rightItems} />
          </div>
          <span
            className={styles.iconButton}
            data-tooltip={themeLabel}
          >
            <NavbarColorModeToggle className={styles.colorModeToggle} />
          </span>
        </div>
      }
    />
  );
}
