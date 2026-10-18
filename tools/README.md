# FYP browser launcher

`start-fyp-browser.ps1` reuses the already installed official Microsoft Playwright
MCP and the dedicated `%LOCALAPPDATA%/CrimeScopeAI/browser-profile`. It installs
nothing and does not access a personal Chrome profile. The default configuration
is `playwright-native.json`: native maximized window, **viewport:null**.

Do not use `browser_resize` or leave `page.setViewportSize()` active in the user's
demonstration tab. An emulated viewport can exceed the visible native window,
making the page's logical bottom appear clipped even when CSS scrolling works.
Responsive QA must use cleanup that opens a new native page and closes the
resized test page. The existing responsive check now does this in `finally`.

The launcher refuses to create a second MCP service on port 8931. Restart only
the verified existing FYP service when changing configuration, preserving the
dedicated persistent profile. `-ConfigPath` permits a separate explicit QA config;
normal use should leave it unset. No forced device scaling is in the defaults.

`tests/native_scroll_checks.js` tests a real 1649×928 native window plus an exact
1649×928 content viewport where display space permits, using mouse-wheel input,
scrollbar dragging and Ctrl+End. It does not emulate viewport dimensions. Its
temporary test geometry is restored to the maximized window in `finally`.
