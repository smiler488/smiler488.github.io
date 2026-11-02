import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import isInternalUrl from "@docusaurus/isInternalUrl";
import { isRegexpStringMatch } from "@docusaurus/theme-common";
import { translate } from "@docusaurus/Translate";
import IconExternalLink from "@theme/Icon/ExternalLink";
import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";
import NavIcon, { navIconName } from "@site/src/components/NavIcon";
import styles from "./styles.module.css";

function NavbarNavLink({
  activeBasePath,
  activeBaseRegex,
  to,
  href,
  label,
  html,
  isDropdownLink,
  prependBaseUrlToHref,
  className,
  ...props
}) {
  const toUrl = useBaseUrl(to);
  const activeBaseUrl = useBaseUrl(activeBasePath);
  const normalizedHref = useBaseUrl(href, { forcePrependBaseUrl: true });
  const isExternalLink = label && href && !isInternalUrl(href);

  // SSR-safe active state check
  const isActive = React.useMemo(() => {
    if (!ExecutionEnvironment.canUseDOM) {
      return false;
    }

    // For docSidebar type (Tutorial), ensure it only activates on /docs/ paths
    if (activeBasePath || activeBaseRegex) {
      // Check if this is a docSidebar navigation item
      if (label === "Tutorial" || label === "tutorial") {
        // Only activate for /docs/ paths, not for other paths that might start with /docs/
        return (
          location.pathname === "/docs/" ||
          location.pathname.startsWith("/docs/tutorial-apps/") ||
          location.pathname.startsWith("/docs/current/")
        );
      }

      // For other docSidebar items, use the original logic
      return (
        isRegexpStringMatch(activeBaseRegex, location.pathname) ||
        location.pathname.startsWith(activeBaseUrl)
      );
    }

    // For regular navigation items (Home, Blog, CV, Resources, App)
    if (toUrl) {
      // Exact match or starts with toUrl + '/'
      return (
        location.pathname === toUrl || location.pathname.startsWith(toUrl + "/")
      );
    }

    return false;
  }, [activeBasePath, activeBaseRegex, activeBaseUrl, toUrl, label]);

  const iconName = navIconName(label);

  return (
    <Link
      {...props}
      {...(href
        ? {
            href: prependBaseUrlToHref ? normalizedHref : href,
          }
        : {
            to: toUrl,
          })}
      className={clsx(
        "menu__link",
        isActive && "menu__link--active",
        styles.mobileNavbarLink,
        className
      )}
    >
      {iconName && (
        <span className={styles.mobileNavbarIcon}>
          <NavIcon
            name={iconName}
            outlineClassName={styles.iconOutline}
            solidClassName={styles.iconSolid}
          />
        </span>
      )}
      <span className={styles.mobileNavbarLabel}>{label}</span>
      {isExternalLink && <IconExternalLink width={12} height={12} />}
    </Link>
  );
}

export default NavbarNavLink;
