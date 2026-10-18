# Native Chrome scrolling correction — 2026-10-18

## Root cause found

A previous responsive automation test left a fixed **1649×928 emulated page
viewport** inside a smaller **1280×680 native Chrome window**. The page could
reach its logical bottom while the rendered footer lay outside the visible
window. In the mixed results state the logical footer ended near y=898, while
that native window's actual content area was only about 537 pixels tall. This
matches a view stopping within/around the third scenario despite a document
scrollbar already at its end.

No CSS overflow lock, fixed-height results wrapper, nested vertical scroller,
hidden third-card content or inference-driven height truncation was found.
Prior full-page screenshots and emulated-viewport checks were insufficient:
they captured the virtual page, not proof of native window visibility.

## Applied correction

- Official Microsoft Playwright MCP now starts with `viewport: null`, using
  `tools/playwright-native.json`. Its content viewport follows the native window.
- The FYP-only local browser launcher points to the versioned
  `tools/start-fyp-browser.ps1`; its previous launcher was backed up locally.
  The same dedicated persistent profile is reused. No cookies/tokens copied.
- The responsive test now uses `finally` cleanup to close the resized test page
  and create a fresh native-sized page, including after failures. It refuses an
  incorrectly configured starting context instead of silently leaving emulation.
- The MCP service was restarted and the default native scaling restored after
  QA. The current browser contains the same one-image/one-video results, with
  wheel scrolling reaching the visible footer.

Website/scenario generation, CSS, scoring, checkpoints and reviews are unchanged.

## Actual tests and screenshots

`tests/native_scroll_checks.js` uses headed Chrome, **no page.setViewportSize**,
and tests native mouse-wheel input, scrollbar-thumb dragging and Ctrl+End both
before upload and after analyzing one real sample-derived PNG + the real MP4.
All 12 checks passed:

| Native test | Content area | Result page height | End scroll Y | Footer bottom |
|---|---|---:|---:|---:|
| Window 1649×928 | 1633×777 (Chrome UI excluded) | 2248 | 1471 | 746.6 |
| Exact CSS viewport 1649×928 | 1649×928 | 2248 | 1320 | 897.6 |

All three methods reached the actual maximum scroll position. The entire third
scenario, collapsed Advanced Details and footer were simultaneously visible.
All checked elements had normal overflow and no nested clipping container.
Before upload, the exact 1649×928 viewport already fits the complete page: no
scrollbar is needed and the footer is visible, rather than artificially forcing
extra height.

For exact-size QA only, Chrome was launched at device scale 1 so those real
window dimensions fit the physical display. This did not change Windows display
settings. The persistent launch configuration does **not** force scaling. After
QA it was restarted at the original native scale 1.5 and maximized window; the
mixed results footer was visible at y=507.6 in its real 537px content height.

- **15 Python incident/configuration tests passed**.
- **20 responsive browser checks passed again**, including native-page cleanup.
- Both JavaScript test files passed Node syntax checks.
- Native scroll measurements: `native_scroll_checks.json`.
- All screenshots below are **viewport captures (`fullPage:false`)**, not full
  page composites. They show the actual scrollbar and bottom of the page:
  - `native-bottom-viewport-1649x928-results-wheel.png`
  - `native-bottom-viewport-1649x928-results-scrollbar-drag.png`
  - `native-bottom-viewport-1649x928-results-ctrl-end.png`
  - `native-bottom-viewport-1649x928-before-upload-ctrl-end.png`
  - Corresponding native-window captures use the `window-1649x928` prefix.
  - `native-bottom-restored-window.png` shows the final normal native-DPI window.

If this occurs in a different Chrome profile that is not controlled by this FYP
MCP, the viewport-emulation diagnosis does not by itself establish that profile's
cause; these measurements and the correction apply to the controlled browser.
