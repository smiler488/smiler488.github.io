import React from "react";
import clsx from "clsx";
import {
  ErrorCauseBoundary,
  ThemeClassNames,
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

function NavbarContentLayout({ left, right }) {
  return (
    <div className={clsx("navbar__inner", styles.navbarSurface)}>
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
            aria-label={
              i18n.currentLocale === "zh-Hans" ? "主要页面" : "Primary pages"
            }
          >
            <NavbarItems items={leftItems} />
          </nav>
        </>
      }
      right={
        <div className={styles.utilityNav}>
          {!hasConfiguredSearch && (
            <div className={styles.searchSlot}>
              <NavbarSearch>
                <SearchBar />
              </NavbarSearch>
            </div>
          )}
          <div className={styles.rightItems}>
            <NavbarItems items={rightItems} />
          </div>
          <NavbarColorModeToggle className={styles.colorModeToggle} />
        </div>
      }
    />
  );
}
