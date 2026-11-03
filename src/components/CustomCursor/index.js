import React from "react";

/**
 * CustomCursor — frosted-glass circular cursor with glow ring.
 *
 * It switches on from the actual input, not from device detection: the dot
 * and ring stay hidden (and the system cursor stays visible) until a mouse
 * pointer moves. Touch or pen input hides them again. This matters on
 * Windows laptops with touch screens, which report touch support and a fine
 * pointer at the same time. Skipped entirely with prefers-reduced-motion.
 *
 * The on-state is a data attribute on <html>, not a class: Docusaurus
 * rewrites the <html> class list on every route change, which used to wipe
 * the cursor after the first navigation.
 */
const ON_ATTR = "data-custom-cursor";

export default function CustomCursor() {
  const dotRef = React.useRef(null);
  const ringRef = React.useRef(null);

  React.useEffect(() => {
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const dot = dotRef.current;
    const ring = ringRef.current;
    if (!dot || !ring) return undefined;
    const root = document.documentElement;

    let mouseX = -100;
    let mouseY = -100;
    let dotX = mouseX;
    let dotY = mouseY;
    let ringX = mouseX;
    let ringY = mouseY;
    let rafId = 0;
    let enabled = false; // a mouse is the current input

    const animate = () => {
      rafId = 0;
      if (!enabled) return;
      // Smooth lerp — dot follows closely, ring trails slightly behind.
      dotX += (mouseX - dotX) * 0.18;
      dotY += (mouseY - dotY) * 0.18;
      ringX += (mouseX - ringX) * 0.1;
      ringY += (mouseY - ringY) * 0.1;
      dot.style.transform = `translate3d(${dotX - 4}px, ${dotY - 4}px, 0)`;
      ring.style.transform = `translate3d(${ringX - 18}px, ${ringY - 18}px, 0)`;
      rafId = requestAnimationFrame(animate);
    };

    const enable = (x, y) => {
      if (enabled || motionQuery.matches) return;
      enabled = true;
      // Start at the pointer, not at the top-left corner.
      dotX = ringX = mouseX = x;
      dotY = ringY = mouseY = y;
      root.setAttribute(ON_ATTR, "");
      if (!rafId) rafId = requestAnimationFrame(animate);
    };

    const disable = () => {
      enabled = false;
      root.removeAttribute(ON_ATTR);
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
    };

    const onPointerMove = (e) => {
      if (e.pointerType !== "mouse") {
        disable();
        return;
      }
      if (!enabled) enable(e.clientX, e.clientY);
      // Restore the flag if anything else rewrote <html> attributes
      else if (!root.hasAttribute(ON_ATTR)) root.setAttribute(ON_ATTR, "");
      mouseX = e.clientX;
      mouseY = e.clientY;
    };

    const onPointerDown = (e) => {
      if (e.pointerType !== "mouse") disable();
    };

    // Leaving the window: park the cursor off-screen.
    const onMouseOut = (e) => {
      if (!e.relatedTarget) {
        mouseX = -100;
        mouseY = -100;
      }
    };

    // Selector for interactive elements that should trigger hover state
    const INTERACTIVE_SEL =
      "a, button, [role='button'], input, select, textarea, label, [tabindex]:not([tabindex='-1']), [class*='card'], [class*='btn']";

    // Decided on every mouseover, so a link that unmounts under the pointer
    // (route change) can't leave the ring stuck in its hover state.
    const onMouseOver = (e) => {
      const over = Boolean(e.target.closest?.(INTERACTIVE_SEL));
      dot.classList.toggle("hovering", over);
      ring.classList.toggle("hovering", over);
    };

    const onHoverOut = (e) => {
      const target = e.target.closest?.(INTERACTIVE_SEL);
      if (target && !(e.relatedTarget && target.contains(e.relatedTarget))) {
        dot.classList.remove("hovering");
        ring.classList.remove("hovering");
      }
    };

    const onMotionChange = (e) => {
      if (e.matches) disable();
    };

    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerdown", onPointerDown, { passive: true });
    document.addEventListener("mouseout", onMouseOut, { passive: true });
    document.addEventListener("mouseover", onMouseOver, { passive: true });
    document.addEventListener("mouseout", onHoverOut, { passive: true });
    motionQuery.addEventListener("change", onMotionChange);

    return () => {
      disable();
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("mouseout", onMouseOut);
      document.removeEventListener("mouseover", onMouseOver);
      document.removeEventListener("mouseout", onHoverOut);
      motionQuery.removeEventListener("change", onMotionChange);
    };
  }, []);

  return (
    <>
      <div className="custom-cursor-dot" ref={dotRef} aria-hidden="true" />
      <div className="custom-cursor-ring" ref={ringRef} aria-hidden="true" />
    </>
  );
}
