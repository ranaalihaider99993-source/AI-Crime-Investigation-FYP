'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const input = $('image-files'), button = $('analyze-images'), previews = $('image-previews');
  let urls = [];
  function clear() { urls.forEach(URL.revokeObjectURL); urls = []; }
  input.addEventListener('change', () => {
    clear(); previews.replaceChildren(); $('image-results').hidden = true; $('image-error').hidden = true; $('image-status').textContent = '';
    const files = [...input.files];
    const valid = files.length > 0 && files.length <= 6 && files.every(f => f.size > 0 && f.size <= 8 * 1024 ** 2 && ['image/jpeg','image/png','image/webp'].includes(f.type));
    button.disabled = !valid;
    if (!valid) { $('image-error').textContent = 'Choose 1–6 JPEG, PNG or WebP images, up to 8 MB each.'; $('image-error').hidden = false; return; }
    files.forEach(file => {
      const figure = document.createElement('figure'), image = document.createElement('img'), caption = document.createElement('figcaption');
      const url = URL.createObjectURL(file); urls.push(url); image.src = url; image.alt = 'Selected image preview'; caption.textContent = file.name;
      figure.append(image, caption); previews.append(figure);
    });
  });
  button.addEventListener('click', async () => {
    button.disabled = true; input.disabled = true; $('image-error').hidden = true; $('image-results').hidden = true; $('image-status').textContent = 'Inspecting locally…';
    try {
      const body = new FormData(); [...input.files].forEach(file => body.append('images', file));
      const response = await fetch('/api/images/analyze', {method:'POST', body});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Inspection failed.');
      const brief = $('image-brief'); brief.replaceChildren();
      for (const [key, title] of [['observations','Measured observations'],['hypotheses','Hypotheses'],['uncertainty','Unknowns and limitations']]) {
        const h = document.createElement('h4'), list = document.createElement('ul'); h.textContent = title;
        data.scenario[key].forEach(text => { const li = document.createElement('li'); li.textContent = text; list.append(li); }); brief.append(h, list);
      }
      const provenance = $('image-provenance'); provenance.replaceChildren();
      data.images.forEach(item => { const p = document.createElement('p'); p.textContent = `${item.image_id}: ${item.width} × ${item.height}, ${item.bytes} bytes; SHA-256 ${item.sha256}`; provenance.append(p); });
      const p = document.createElement('p'); p.textContent = data.preprocessing; provenance.append(p);
      $('image-results').hidden = false; $('image-status').textContent = `${data.images.length} image(s) inspected. No semantic image model was run.`;
    } catch (error) { $('image-error').textContent = error.message; $('image-error').hidden = false; $('image-status').textContent = ''; }
    finally { input.disabled = false; button.disabled = input.files.length === 0; }
  });
  window.addEventListener('pagehide', clear);
})();
