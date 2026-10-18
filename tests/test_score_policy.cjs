const test = require('node:test');
const assert = require('node:assert/strict');
const {assess} = require('../web/static/score-policy.js');
const scores = values => values.map((score, i) => ({label:['Burglary','Stealing','Fighting','Vandalism'][i],score}));
test('real sample is inconclusive for weak and close scores', () => {
  const result=assess(scores([.2641257942,.2547009289,.2298146337,.2513586283]));
  assert.equal(result.inconclusive,true); assert.equal(result.weak,true); assert.equal(result.close,true);
  assert.equal(result.top.label,'Burglary');
});
test('a strong separated score can show its top category', () => {
  assert.equal(assess(scores([.7,.15,.1,.05])).inconclusive,false);
});
test('a strong but close top pair remains inconclusive', () => {
  const result=assess(scores([.51,.45,.03,.01]));
  assert.equal(result.weak,false); assert.equal(result.close,true); assert.equal(result.inconclusive,true);
});
test('weak scores remain inconclusive even when separated', () => {
  const result=assess(scores([.49,.2,.19,.12]));
  assert.equal(result.weak,true); assert.equal(result.close,false);
});
test('threshold equality passes; just below either boundary does not', () => {
  assert.equal(assess(scores([.5,.4,.06,.04])).inconclusive,false);
  assert.equal(assess(scores([.4999,.3,.15,.0501])).inconclusive,true);
  assert.equal(assess(scores([.5,.4001,.0599,.04])).inconclusive,true);
});
test('ties are inconclusive and preserve input/class order', () => {
  const input=scores([.25,.25,.25,.25]); const original=JSON.stringify(input);
  assert.equal(assess(input).inconclusive,true); assert.equal(JSON.stringify(input),original);
});
test('malformed scores fail closed', () => {
  for (const input of [[],scores([NaN,.2,.2,.2]),scores([.5,.4,.2,-.1]),scores([.7,.2,.2,.2])]) {
    assert.throws(()=>assess(input),/Invalid model scores/);
  }
});
