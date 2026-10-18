'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const input = $('image-files'), button = $('analyze-images'), previews = $('image-previews');
  fetch('/api/images/model').then(r => {if (!r.ok) throw new Error(); return r.json();}).then(data => {
    $('image-model-status').textContent = data.available ? 'Separate weak-label image CNN verified. CPU inference; no validated scene understanding.' : 'Image checkpoint unavailable: technical inspection only.';
  }).catch(() => { $('image-model-status').textContent = 'Image model status unavailable. No semantic analysis claim.'; });
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
      CrimeScopeNarrative.render($('image-brief'), data.narrative);
      const predictions = $('image-predictions'); predictions.replaceChildren();
      for (const image of data.images) {
        const output = image.experimental_classification; if (!output) continue;
        const decision = CrimeScopeScorePolicy.assess(output.scores);
        const heading = document.createElement('h3'), explanation = document.createElement('p');
        heading.textContent = `${image.image_id}: ${decision.inconclusive ? 'Inconclusive / Low confidence' : decision.top.label}`;
        explanation.textContent = `Highest weak category: ${decision.top.label} (${(decision.top.score*100).toFixed(1)}% relative score). Not a crime probability. No Normal/Unknown class.`;
        const details = document.createElement('details'), summary = document.createElement('summary'); summary.textContent = 'View all model scores'; details.append(summary);
        output.scores.forEach(score => { const p = document.createElement('p'); p.textContent = `${score.label}: ${(score.score*100).toFixed(2)}% uncalibrated relative score`; details.append(p); });
        const policy = document.createElement('p'); policy.className = 'threshold-note'; policy.textContent = 'Experimental display heuristic: inconclusive if top score <50% or top-two margin <10 percentage points. Not calibrated or validated.';
        const provenance = document.createElement('p'); provenance.className = 'privacy'; provenance.textContent = `Image model ${output.model.architecture}; checkpoint SHA-256 ${output.model.checkpoint_sha256}.`;
        predictions.append(heading, explanation, details, policy, provenance);
      }
      const provenance = $('image-provenance'); provenance.replaceChildren();
      data.images.forEach(item => { const p = document.createElement('p'); p.textContent = `${item.image_id}: ${item.width} × ${item.height}, ${item.bytes} bytes; SHA-256 ${item.sha256}`; provenance.append(p); });
      const p = document.createElement('p'); p.textContent = data.preprocessing; provenance.append(p);
      $('image-results').hidden = false; $('image-status').textContent = `${data.images.length} image(s) inspected. ${data.image_model_available ? 'UNVALIDATED weak-label image CNN executed; no scene facts inferred.' : 'No image classification model was run.'}`;
    } catch (error) { $('image-error').textContent = error.message; $('image-error').hidden = false; $('image-status').textContent = ''; }
    finally { input.disabled = false; button.disabled = input.files.length === 0; }
  });
  window.addEventListener('pagehide', clear);
})();
