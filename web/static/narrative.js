'use strict';
window.CrimeScopeNarrative = {
  render(container, narrative) {
    container.replaceChildren();
    const heading = document.createElement('h3'); heading.textContent = 'Shared experimental scenario analysis';
    const warning = document.createElement('p'); warning.className = 'result-warning'; warning.textContent = narrative.warning;
    container.append(heading, warning);
    const sections = [
      ['Visible observations', narrative.visible_observations.length ? narrative.visible_observations.map(x => x.text) : ['No semantic observations verified.']],
      ['Technical evidence', narrative.technical_observations.map(x => `${x.evidence_id}: ${x.text}`)],
      ['Possible incident sequence — abstained', [narrative.possible_incident_sequence.text]],
      ['Crime hypothesis — abstained', [narrative.crime_hypothesis.text]],
      ['Uncertainties', narrative.uncertainties]
    ];
    for (const [title, entries] of sections) {
      const h = document.createElement('h4'), list = document.createElement('ul'); h.textContent = title;
      for (const text of entries) { const li = document.createElement('li'); li.textContent = text; list.append(li); }
      container.append(h, list);
    }
  }
};
