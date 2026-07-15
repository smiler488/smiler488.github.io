---
slug: macOS-shortcuts
title: macOS Keyboard Shortcuts — A Practical Reference
description: A compact, safety-aware reference for everyday macOS, Finder, text-editing, browser, screenshot, and recovery shortcuts.
authors: [liangchao]
category: Developer tools
article_type: Reference
tags: [productivity, reproducible-research]
image: /img/blog-default.jpg
---

## At a glance

These shortcuts cover the actions most useful in research and development work: finding files, navigating text, managing windows, capturing evidence, and recovering from an unresponsive app.

- Shortcuts can vary by app and keyboard layout.
- On newer keyboards, the **Fn/Globe** key may open the Character Viewer.
- Recovery shortcuts can discard unsaved work; read their warnings before using them.

<!-- truncate -->

## Modifier-key legend

| Symbol      | Key              |
| ----------- | ---------------- |
| `⌘`         | Command          |
| `⌥`         | Option / Alt     |
| `⌃`         | Control          |
| `⇧`         | Shift            |
| `Fn` / `🌐` | Function / Globe |

## Search, apps, and windows

- `⌘ Space` — open Spotlight.
- `⌘ Tab` — switch to the next open app; keep holding Command to choose an app.
- `⌘ H` — hide all windows of the front app.
- `⌥ ⌘ H` — hide windows of every app except the front app.
- `⌘ M` — minimize the front window.
- `⌘ W` — close the front window or tab.
- `⌘ Q` — quit the front app.
- `⌃ ⌘ Q` — lock the screen.

## Screenshots and screen recording

- `⇧ ⌘ 3` — capture the entire screen.
- `⇧ ⌘ 4` — capture a selected region; press Space after invoking it to capture a window.
- `⇧ ⌘ 5` — open screenshot and screen-recording controls.

Use `⌃` with a screenshot shortcut when you want to copy the capture to the clipboard instead of saving a file.

## Finder

### Files

- `⌘ Delete` — move selected items to the Trash.
- `⇧ ⌘ Delete` — empty the Trash after confirmation.
- `⌘ I` — show information for the selected item.
- `⌘ D` — duplicate the selected item.
- `Space` — preview the selected item with Quick Look.

### Navigation

- `⌘ ↑` — open the enclosing folder.
- `⌘ ↓` — open the selected item.
- `⇧ ⌘ G` — go to a folder by path.
- `⌘ 1`, `⌘ 2`, `⌘ 3`, `⌘ 4` — switch among icon, list, column, and gallery views.

## Text editing and navigation

These work in most native text fields and many editors:

- `⌘ A` — select all.
- `⌘ C`, `⌘ X`, `⌘ V` — copy, cut, and paste.
- `⌘ Z` — undo.
- `⇧ ⌘ Z` — redo in apps that follow the standard convention.
- `⌥ ←` / `⌥ →` — move one word backward or forward.
- `⌘ ←` / `⌘ →` — move to the beginning or end of the current line.
- `Fn Delete` — forward delete on compact keyboards.

## Safari and Chrome

- `⌘ T` — open a new tab.
- `⌘ W` — close the current tab.
- `⇧ ⌘ T` — reopen the most recently closed tab.
- `⌘ L` — focus the address bar.
- `⌘ R` — reload the page.
- `⌘ [` / `⌘ ]` — go backward or forward in tab history.

Browser extensions and web apps may override some shortcuts.

## Recovery and power controls

:::warning Save work first
The following actions can close apps or discard unsaved changes. Use them only when the normal app or Apple menu controls do not work.
:::

- `⌥ ⌘ Esc` — open Force Quit Applications.
- Press and hold the power button — force the Mac to turn off when it is unresponsive.
- `⌃ ⌘ Power` — on supported built-in keyboards without Touch ID, force a restart without prompting to save open documents.
- `⌃ ⌥ ⌘ Power` — on supported built-in keyboards without Touch ID, ask apps to quit and then shut down; apps with unsaved documents may prompt first.

Power-key behavior differs across Touch ID keyboards, external keyboards, and macOS versions. Prefer **Apple menu → Shut Down** or **Restart** whenever the system still responds.

## Official reference

Apple maintains the authoritative and most current list at [Mac keyboard shortcuts](https://support.apple.com/en-us/102650).
