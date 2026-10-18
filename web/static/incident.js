'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const input = $('incident-files'), button = $('analyze-incident'), error = $('incident-error');
  let selected = [], urls = [], busy = false;
  function fail(message) { error.textContent = message; error.hidden = false; }
  function clearUrls() { urls.forEach(URL.revokeObjectURL); urls = []; }
  function kind(file) {
    if (/\.(jpe?g|png|webp)$/i.test(file.name)) return 'image';
    if (/\.(mp4|avi|mov|mkv|webm)$/i.test(file.name)) return 'video';
    return null;
  }
  function choose(files) {
    if (busy) return;
    clearUrls(); selected = []; button.disabled = true; $('incident-previews').replaceChildren(); $('incident-results').hidden = true; error.hidden = true; $('incident-progress').textContent = '';
    const images = files.filter(f => kind(f) === 'image'), videos = files.filter(f => kind(f) === 'video');
    if (!files.length) return;
    if (images.length + videos.length !== files.length) { fail('Use JPEG, PNG, WebP, MP4, AVI, MOV, MKV or WebM.'); return; }
    if (images.length > 6 || videos.length > 3 || files.some(f => !f.size || (kind(f) === 'image' && f.size > 8*1024**2)) || files.reduce((n,f) => n+f.size,0) > 60*1024**2) {
      fail('Choose at most 6 images and 3 videos; 8 MB per image and 60 MB total. Files must not be empty.'); return;
    }
    selected = files; button.disabled = false;
    files.forEach((file,i) => {
      const figure = document.createElement('figure'), media = document.createElement(kind(file) === 'image' ? 'img' : 'video'), caption = document.createElement('figcaption');
      const url = URL.createObjectURL(file); urls.push(url); media.src = url;
      if (kind(file) === 'image') media.alt = 'Selected evidence preview';
      else { media.controls = true; media.preload = 'metadata'; }
      caption.textContent = `file-${i+1}: ${file.name}`; figure.append(media,caption); $('incident-previews').append(figure);
    });
  }
  input.addEventListener('change', () => choose([...input.files]));
  for (const type of ['dragover','dragleave','drop']) $('incident-drop').addEventListener(type,event => {
    event.preventDefault(); $('incident-drop').classList.toggle('drag',type === 'dragover'); if (type === 'drop') choose([...event.dataTransfer.files]);
  });
  button.addEventListener('click', async () => {
    if (busy || !selected.length) return;
    busy = true; input.disabled = true; button.disabled = true; error.hidden = true; $('incident-results').hidden = true;
    $('incident-progress').textContent = 'Analyzing files sequentially on local CPU…';
    try {
      const body = new FormData(); selected.forEach(file => body.append('evidence',file));
      const response = await fetch('/api/incidents/analyze',{method:'POST',body});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Incident analysis failed.');
      $('incident-status').textContent = data.status;
      const scenarios = $('incident-scenarios'); scenarios.replaceChildren();
      data.scenarios.forEach(item => {
        const article = document.createElement('article'), heading = document.createElement('h3'), score = document.createElement('strong'), explanation = document.createElement('p');
        article.className = 'scenario-card'; heading.textContent = `${item.rank}. ${item.title}`; score.textContent = `${(item.relative_score*100).toFixed(1)}% relative score`; explanation.textContent = item.explanation;
        article.append(heading,score,explanation); scenarios.append(article);
      });
      const uncertainties = $('incident-uncertainties'); uncertainties.replaceChildren();
      data.uncertainties.forEach(text => { const li = document.createElement('li'); li.textContent = text; uncertainties.append(li); });
      const technical = $('incident-technical'); technical.replaceChildren();
      for (const text of [data.method,data.display_rule,data.ranking_note,`${data.unique_files} unique file(s); ${data.duplicates_ignored} exact duplicate upload(s) ignored.`]) {
        const p = document.createElement('p'); p.textContent = text; technical.append(p);
      }
      const pre = document.createElement('pre'); pre.textContent = JSON.stringify({scores:data.scores,modality_means:data.modality_means,files:data.files},null,2); technical.append(pre);
      $('incident-advanced').open = false; $('incident-results').hidden = false; $('incident-progress').textContent = 'Analysis complete. Unverified hypotheses only; no incident facts established.';
    } catch (failure) { fail(failure.message); $('incident-progress').textContent = ''; }
    finally { busy = false; input.disabled = false; button.disabled = !selected.length; }
  });
  window.addEventListener('pagehide',clearUrls);
})();
