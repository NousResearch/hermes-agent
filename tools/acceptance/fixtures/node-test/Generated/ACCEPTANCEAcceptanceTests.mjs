import test from 'node:test';

// [acceptance:metadata] {"PathCode":"UT-ACCEPTANCE-001","Capability":"acceptance-test-workflow","Fingerprint":"5e6d5b63c9e6e4e0aefea4099b6d7d81bb8fd0f9182a872dc1a80f5cc057a219","Ignored":false}
test("UT-ACCEPTANCE-001 Generated acceptance test preserves the developer-owned body across regeneration", async () => {
  // [acceptance:body:UT-ACCEPTANCE-001]
  const preserved = 'developer-owned body';
  if (preserved !== 'developer-owned body') throw new Error('body changed');
  // [/acceptance:body:UT-ACCEPTANCE-001]
});
