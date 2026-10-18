// Run in the existing headed MCP context configured with viewport:null.
// No page.setViewportSize: it can leave a virtual viewport beyond the real window.
async (page) => {
  const cdp=await page.context().newCDPSession(page);
  const {windowId}=await cdp.send('Browser.getWindowForTarget');
  const checks=[];
  const assert=(ok,message)=>{if(!ok)throw new Error(message);};
  async function metrics() {
    return page.evaluate(()=>{
      const root=document.scrollingElement, footer=document.querySelector('footer').getBoundingClientRect();
      const third=document.querySelectorAll('.scenario-card')[2]?.getBoundingClientRect();
      const advanced=document.querySelector('#incident-advanced').getBoundingClientRect();
      return {inner:[innerWidth,innerHeight],outer:[outerWidth,outerHeight],screen:[screen.width,screen.height],dpr:devicePixelRatio,
        scrollY,maximum:Math.max(0,root.scrollHeight-innerHeight),documentHeight:root.scrollHeight,
        footer:{top:footer.top,bottom:footer.bottom,visible:footer.top>=0&&footer.bottom<=innerHeight+1},
        third:third?{top:third.top,bottom:third.bottom,visible:third.top>=0&&third.bottom<=innerHeight+1}:null,
        advanced:{top:advanced.top,bottom:advanced.bottom},
        noHorizontalOverflow:root.scrollWidth<=innerWidth,
        clipped:[...document.querySelectorAll('body *')].filter(e=>['hidden','clip','auto','scroll'].includes(getComputedStyle(e).overflowY)&&e.scrollHeight>e.clientHeight+1).map(e=>({tag:e.tagName,id:e.id,overflow:getComputedStyle(e).overflowY})),
        rootOverflow:getComputedStyle(root).overflowY,bodyOverflow:getComputedStyle(document.body).overflowY};
    });
  }
  async function testBottom(state,sizeLabel,method) {
    await page.evaluate(()=>window.scrollTo(0,0));
    if(method==='wheel') {
      const {width,height}=await page.evaluate(()=>({width:innerWidth,height:innerHeight}));
      await page.mouse.move(width/2,height/2); await page.mouse.wheel(0,100000);
    } else if(method==='ctrl-end') {
      await page.keyboard.press('Control+End');
    } else {
      const bar=await page.evaluate(()=>{const r=document.scrollingElement;return {x:(innerWidth+r.clientWidth)/2,thumb:innerHeight*innerHeight/r.scrollHeight,height:innerHeight,needed:r.scrollHeight>innerHeight+1};});
      if(bar.needed) {
        await page.mouse.move(bar.x,Math.max(8,bar.thumb/2));await page.mouse.down();
        await page.mouse.move(bar.x,bar.height-2,{steps:12});await page.mouse.up();
      }
    }
    await page.waitForTimeout(300);
    const result=await metrics();
    assert(result.footer.visible,`${state}/${sizeLabel}/${method}: footer not visible`);
    assert(Math.abs(result.scrollY-result.maximum)<2,`${state}/${sizeLabel}/${method}: not at actual bottom`);
    assert(result.noHorizontalOverflow,`${state}/${sizeLabel}: horizontal clipping`);
    if(state==='results') {
      assert(result.third?.visible,'Entire third scenario must fit above the actual footer');
      assert(result.advanced.top>=0&&result.advanced.bottom<=result.inner[1],'Advanced Details not reachable');
    }
    checks.push({state,sizeLabel,method,...result});
    await page.screenshot({path:`reports/native-bottom-${sizeLabel}-${state}-${method}.png`,fullPage:false});
  }
  async function runState(sizeLabel) {
    await page.goto('http://127.0.0.1:8000/');
    for(const method of ['wheel','scrollbar-drag','ctrl-end'])await testBottom('before-upload',sizeLabel,method);
    await page.locator('#incident-files').setInputFiles([
      'D:/AI-Crime-Investigation-FYP/artifacts/inference/real-frame.png',
      'D:/AI-Crime-Investigation-FYP/artifacts/inference/sample.mp4']);
    await page.locator('#analyze-incident').click();await page.locator('#incident-results').waitFor({state:'visible'});
    assert(await page.locator('.scenario-card').count()===3,'Mixed analysis must produce three scenarios');
    for(const method of ['wheel','scrollbar-drag','ctrl-end'])await testBottom('results',sizeLabel,method);
  }
  try {
    assert(page.viewportSize()===null,'Native viewport:null is required; refuse a misleading emulated-window test');
    await page.keyboard.press('Control+0');
    await cdp.send('Browser.setWindowBounds',{windowId,bounds:{windowState:'normal'}});
    await cdp.send('Browser.setWindowBounds',{windowId,bounds:{left:0,top:0,width:1649,height:928}});
    await page.waitForTimeout(200);
    await runState('window-1649x928');
    // Size the real window to fit an exact 1649×928 CSS content area;
    // no device-metrics override or page.setViewportSize is used.
    for(let i=0;i<5;i++) {
      const m=await metrics();const {bounds}=await cdp.send('Browser.getWindowBounds',{windowId});
      if(m.inner[0]===1649&&m.inner[1]===928)break;
      const scale=m.dpr;
      await cdp.send('Browser.setWindowBounds',{windowId,bounds:{width:Math.round(bounds.width+(1649-m.inner[0])*scale),height:Math.round(bounds.height+(928-m.inner[1])*scale)}});
      await page.waitForTimeout(200);
    }
    const exact=await metrics();assert(exact.inner[0]===1649&&exact.inner[1]===928,`Exact native viewport unavailable: ${exact.inner}`);
    await runState('viewport-1649x928');
    return {passed:true,checks};
  } finally {
    await page.keyboard.press('Control+0');
    await cdp.send('Browser.setWindowBounds',{windowId,bounds:{windowState:'maximized'}});
    await cdp.detach();
  }
}
