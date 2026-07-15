---
title: Research Maze
description: "A compact canvas experiment in procedural maze generation, keyboard and touch input, movement trails and local scores."
sidebar_label: Maze
sidebar_position: 14
hide_title: true
keywords: [maze, canvas, game, physics, keyboard]
app_route: /app/maze
app_icon: "2D"
app_category: Utilities
app_runtime: "Local canvas experiment"
app_tone: orange
app_badges: ["Canvas", "Keyboard + touch", "Local scores"]
---

## What it does

Research Maze is a local Canvas experiment that generates a new 21 × 21 maze, accepts keyboard or on-page direction controls, draws the player's trail, times the run, and keeps the best ten results in the current browser.

:::info Local experiment

The maze, movement, timer, and leaderboard run in the browser. Scores are stored locally and are never submitted to an online leaderboard.

:::

## Before you start

- On desktop, click outside the player-name field before using the arrow keys.
- On touch devices, use the four large direction buttons below the board.
- Enter a player name of up to 32 characters before finishing if you want it saved with the result.
- A new random maze can be easier or harder than the previous one; records are not normalized for maze difficulty.

## Quick workflow

1. Enter a name in **Player**.
2. Select **New Game** to generate a fresh maze and reset the current timer, steps, trail, and victory message.
3. Move with the keyboard arrow keys or the ▲, ▼, ◀, and ▶ controls.
4. Follow the open white cells from the upper-left start to the right-edge exit. The exit uses a hamburger image when the asset loads and a green fallback otherwise.
5. On completion, review elapsed time and steps in the victory message and **Leaderboard**.
6. Select **New Game** to try another procedurally generated board.

## Controls & outputs

| Control or output | Current behaviour                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| Player            | Sets the name saved with a completed run; empty names become `Player`.                         |
| New Game          | Generates a new 21 × 21 maze and resets the current run.                                       |
| Arrow keys        | Attempts one-cell movement when focus is not in an input, button, select, or editable element. |
| Direction buttons | Provide the same one-cell movement for mouse and touch input.                                  |
| Steps             | Counts successful moves only; attempts into walls do not increment it.                         |
| Trail             | Marks previously occupied cells, keeping at most 500 trail entries.                            |
| Victory message   | Reports the name, elapsed seconds, and successful moves.                                       |
| Leaderboard       | Shows up to 10 local results, sorted first by time and then by steps.                          |

## How it works

Each new game fills a 21 × 21 grid with walls and uses a randomized recursive-backtracker algorithm to carve connected paths from grid position `(1, 1)`. The exit is opened on the right edge and connected to the carved maze.

The timer starts on the first direction attempt, including an attempt blocked by a wall. Only successful moves update **Steps** and the trail. Reaching the exit ends the run, rounds elapsed time to two decimal places, and inserts the result into the sorted local top ten.

## Data, privacy & external services

- Leaderboard entries are stored in browser `localStorage` under `mazeRankings`.
- No name, score, movement path, or device data is uploaded.
- Clearing this site's browser storage removes the leaderboard.
- The game loads the Einstein player and hamburger exit images from this website; simple drawn fallbacks are used when those assets fail.

## Limitations

- Movement is grid-based. The app does not use device tilt, accelerometer input, continuous marble physics, or multiplayer networking.
- Local scores are specific to the browser profile and device; they do not sync across browsers.
- Random mazes vary in route length and difficulty, so leaderboard times are informal rather than directly controlled experimental comparisons.
- The Canvas provides a concise accessible label, but it does not expose every wall and path as navigable screen-reader elements.
- There is no in-app control to clear all saved rankings; clear the site's browser storage if required.

## Troubleshooting

| Problem                                       | What to check                                                                                             |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Arrow keys do not move                        | Move focus out of the **Player** field or a button, then try again; on mobile use the direction controls. |
| A direction press does not increase Steps     | The attempted cell is a wall; only valid moves count.                                                     |
| The timer started before the first valid move | Timing begins with the first direction attempt, even if that attempt is blocked.                          |
| The exit image is missing                     | Use the green exit fallback; gameplay is unchanged.                                                       |
| A completed score is absent                   | Browser storage may be blocked or full, or the result may fall outside the fastest ten.                   |
| You want to remove rankings                   | Clear site data for this origin in the browser settings.                                                  |

[Open Research Maze](/app/maze)

[Back to App Lab](/app)
