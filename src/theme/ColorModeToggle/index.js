import React from "react";
import clsx from "clsx";
import useIsBrowser from "@docusaurus/useIsBrowser";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import IconLightMode from "@theme/Icon/LightMode";
import IconDarkMode from "@theme/Icon/DarkMode";
import IconSystemColorMode from "@theme/Icon/SystemColorMode";
import styles from "./styles.module.css";

/**
 * Theme switch as an icon button with a small menu: follow the system, light
 * or dark. A cycling toggle hides "system" behind two clicks; the menu shows
 * all three and marks the current one. Same props as the stock component, so
 * the navbar and the mobile sidebar both use it.
 */
const OPTIONS = [
  { value: null, key: "system", Icon: IconSystemColorMode },
  { value: "light", key: "light", Icon: IconLightMode },
  { value: "dark", key: "dark", Icon: IconDarkMode },
];

const LABELS = {
  en: { menu: "Theme", system: "System", light: "Light", dark: "Dark" },
  zh: { menu: "主题", system: "跟随系统", light: "浅色", dark: "深色" },
};

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="m5 12.5 4.5 4.5L19 7.5" />
    </svg>
  );
}

function ColorModeToggle({ className, buttonClassName, value, onChange }) {
  const isBrowser = useIsBrowser();
  const { i18n } = useDocusaurusContext();
  const labels = i18n.currentLocale === "zh-Hans" ? LABELS.zh : LABELS.en;
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef(null);
  const buttonRef = React.useRef(null);
  const menuId = React.useId();
  const currentKey = value ?? "system";

  const close = React.useCallback((returnFocus) => {
    setOpen(false);
    if (returnFocus) buttonRef.current?.focus();
  }, []);

  // Close on a press outside or on Escape
  React.useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (e) => {
      if (!rootRef.current?.contains(e.target)) close(false);
    };
    const onKeyDown = (e) => {
      if (e.key === "Escape") close(true);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, close]);

  // Opening moves focus to the current choice; arrows move between choices
  React.useEffect(() => {
    if (open) {
      rootRef.current?.querySelector('[aria-checked="true"]')?.focus();
    }
  }, [open]);

  const onMenuKeyDown = (e) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = [...rootRef.current.querySelectorAll('[role="menuitemradio"]')];
    const step = e.key === "ArrowDown" ? 1 : -1;
    const next = (items.indexOf(document.activeElement) + step + items.length) % items.length;
    items[next].focus();
  };

  const choose = (optionValue) => {
    onChange(optionValue);
    close(true);
  };

  return (
    <div ref={rootRef} className={clsx(styles.root, className)}>
      <button
        ref={buttonRef}
        className={clsx("clean-btn", styles.button, buttonClassName)}
        type="button"
        disabled={!isBrowser}
        aria-label={`${labels.menu}: ${labels[currentKey]}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((v) => !v)}
      >
        {/* All three icons render; html[data-theme-choice] picks one, which
            also works before hydration */}
        <IconSystemColorMode aria-hidden className={clsx(styles.icon, styles.iconSystem)} />
        <IconLightMode aria-hidden className={clsx(styles.icon, styles.iconLight)} />
        <IconDarkMode aria-hidden className={clsx(styles.icon, styles.iconDark)} />
      </button>

      {open && (
        <div
          id={menuId}
          className={styles.menu}
          role="menu"
          aria-label={labels.menu}
          onKeyDown={onMenuKeyDown}
        >
          {OPTIONS.map(({ value: optionValue, key, Icon }) => (
            <button
              key={key}
              type="button"
              role="menuitemradio"
              aria-checked={key === currentKey}
              className={styles.item}
              onClick={() => choose(optionValue)}
            >
              <Icon aria-hidden className={styles.itemIcon} />
              <span className={styles.itemLabel}>{labels[key]}</span>
              <span className={styles.check}>
                <CheckIcon />
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default React.memo(ColorModeToggle);
