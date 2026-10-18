// Run through the existing Playwright MCP browser_run_code_unsafe filename tool.
// No second browser, external services or dataset downloads. Local QA media only.
async (page) => {
  const base = 'D:/AI-Crime-Investigation-FYP/artifacts/inference/';
  const cases = {
    images: [base+'real-frame.png'],
    videos: [base+'sample.mp4', base+'sample-alternate.avi'],
    mixed: [base+'real-frame.png', base+'sample.mp4']
  };
  const checks = [];
  const assert = (condition,message) => {if (!condition) throw new Error(message);};
  assert(page.viewportSize()===null,'Start this responsive check using tools/playwright-native.json');
  async function bottom(stage,viewport,mode) {
    await page.mouse.move(viewport.width/2,viewport.height/2);
    await page.mouse.wheel(0,100000);
    // Chrome applies wheel input asynchronously; inspect its settled position.
    await page.waitForTimeout(250);
    const result = await page.evaluate(() => {
      const root=document.scrollingElement, footer=document.querySelector('footer').getBoundingClientRect();
      return {scrollY,maximum:Math.max(0,root.scrollHeight-innerHeight),footerVisible:footer.bottom<=innerHeight+1 && footer.top>=0,
        noHorizontalOverflow:root.scrollWidth<=innerWidth,
        pageHeight:root.scrollHeight,viewportHeight:innerHeight,
        nestedScrollers:[...document.querySelectorAll('main,section,article')].filter(e =>
          ['auto','scroll','hidden','clip'].includes(getComputedStyle(e).overflowY) && e.scrollHeight>e.clientHeight+1).length};
    });
    assert(result.footerVisible,`${stage}/${mode}: footer clipped`);
    assert(result.noHorizontalOverflow,`${stage}/${mode}: horizontal overflow`);
    assert(Math.abs(result.scrollY-result.maximum)<=2,`${stage}/${mode}: wheel did not reach actual bottom`);
    assert(result.nestedScrollers===0,`${stage}/${mode}: nested scrolling/clipping`);
    checks.push({viewport:`${viewport.width}x${viewport.height}`,mode,stage,...result});
  }
  try {
  for (const viewport of [{width:1280,height:720},{width:390,height:844}]) {
    await page.setViewportSize(viewport);
    for (const [mode,files] of Object.entries(cases)) {
      await page.goto('http://127.0.0.1:8000/');
      assert(await page.locator('#analyze-incident').isDisabled(),'No-upload button must be disabled');
      assert(await page.locator('#incident-results').isHidden(),'No-upload results must be hidden');
      await bottom('before-upload',viewport,mode);
      await page.locator('#incident-files').setInputFiles(files);
      assert(await page.locator('#incident-previews figure').count()===files.length,'Previews missing');
      await bottom('selected',viewport,mode);
      await page.locator('#analyze-incident').click();
      await page.locator('#incident-results').waitFor({state:'visible'});
      assert(await page.locator('.scenario-card').count()===3,'Three readable scenarios required');
      assert(await page.locator('.scenario-uncertainty').count()===3,'Scenario-specific uncertainty missing');
      assert(await page.locator('.scenario-basis').count()===3,'Model/fiction distinction missing');
      assert(!(await page.locator('#incident-advanced').evaluate(e=>e.open)),'Advanced must start collapsed');
      await bottom('results',viewport,mode);
      if (mode==='mixed') {
        await page.screenshot({path:`reports/scenarios-scroll-${viewport.width}.png`,fullPage:true});
        await page.locator('#incident-advanced summary').click();
        await bottom('advanced-open',viewport,mode);
        await page.locator('#incident-advanced summary').click();
      }
    }
  }
  return {passed:true,checks};
  } finally {
    // A new page inherits viewport:null from the context. A resized test page
    // otherwise keeps its virtual size and can clip the user's native window.
    const nativePage=await page.context().newPage();
    await nativePage.goto('http://127.0.0.1:8000/');
    await page.close();
  }
}
