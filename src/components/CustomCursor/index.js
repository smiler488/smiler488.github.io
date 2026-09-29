import React from "react";

/**
 * CustomCursor — frosted-glass circular cursor with glow ring.
 * Renders a soft blurred dot + an outer glow ring that follows the mouse.
 * Hidden on touch devices and when prefers-reduced-motion is set.
 */
export default function CustomCursor() {
  const dotRef = React.useRef(null);
  const ringRef = React.useRef(null);

  React.useEffect(() => {
    // Detect touch device — skip custom cursor on touch
    const hasTouch =
      "ontouchstart" in window || navigator.maxTouchPoints > 0;
    if (hasTouch) return undefined;

    // Respect reduced-motion preference
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (motionQuery.matches) return undefined;

    const dot = dotRef.current;
    const ring = ringRef.current;
    if (!dot || !ring) return undefined;

    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    let dotX = mouseX;
    let dotY = mouseY;
    let ringX = mouseX;
    let ringY = mouseY;
    let rafId = 0;
    let active = true;

    const onMouseMove = (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };

    const onMouseLeave = () => {
      // Move cursor off-screen when mouse leaves the window
      mouseX = -100;
      mouseY = -100;
    };

    const animate = () => {
      if (!active) return;

      // Smooth lerp — dot follows closely, ring trails slightly behind
      dotX += (mouseX - dotX) * 0.18;
      dotY += (mouseY - dotY) * 0.18;
      ringX += (mouseX - ringX) * 0.1;
      ringY += (mouseY - ringY) * 0.1;

      dot.style.transform = `translate3d(${dotX - 4}px, ${dotY - 4}px, 0)`;
      ring.style.transform = `translate3d(${ringX - 18}px, ${ringY - 18}px, 0)`;

      rafId = requestAnimationFrame(animate);
    };

    // Selector for interactive elements that should trigger hover state
    const INTERACTIVE_SEL =
      "a, button, [role='button'], input, select, textarea, label, [tabindex]:not([tabindex='-1']), [class*='card'], [class*='btn']";

    const onMouseOver = (e) => {
      const target = e.target.closest(INTERACTIVE_SEL);
      if (target) {
        dot.classList.add("hovering");
        ring.classList.add("hovering");
      }
    };

    const onMouseOut = (e) => {
      const target = e.target.closest(INTERACTIVE_SEL);
      if (target) {
        // Only remove if we're truly leaving the interactive element
        const related = e.relatedTarget;
        if (!related || !target.contains(related)) {
          dot.classList.remove("hovering");
          ring.classList.remove("hovering");
        }
      }
    };

    document.addEventListener("mouseover", onMouseOver, { passive: true });
    document.addEventListener("mouseout", onMouseOut, { passive: true });

    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("mouseleave", onMouseLeave, { passive: true });
    rafId = requestAnimationFrame(animate);

    const onMotionChange = (e) => {
      if (e.matches) {
        active = false;
        cancelAnimationFrame(rafId);
        dot.style.opacity = "0";
        ring.style.opacity = "0";
      } else {
        active = true;
        dot.style.opacity = "";
        ring.style.opacity = "";
        rafId = requestAnimationFrame(animate);
      }
    };
    motionQuery.addEventListener("change", onMotionChange);

    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseleave", onMouseLeave);
      document.removeEventListener("mouseover", onMouseOver);
      document.removeEventListener("mouseout", onMouseOut);
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
