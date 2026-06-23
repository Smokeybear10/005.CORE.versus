const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function harness(filename, fetch) {
    const elements = new Map();
    const listeners = new Map();
    const events = [];
    const element = id => {
        if (!elements.has(id)) elements.set(id, {
            hidden: false, disabled: false, value: '', innerHTML: '', textContent: '',
            listeners: new Map(), attributes: new Map(),
            classList: {add() {}, remove() {}},
            addEventListener(name, callback) { this.listeners.set(name, callback); },
            setAttribute(name, value) { this.attributes.set(name, value); },
            append() {},
        });
        return elements.get(id);
    };
    const window = {
        addEventListener(name, callback) {
            if (!listeners.has(name)) listeners.set(name, []);
            listeners.get(name).push(callback);
        },
        dispatchEvent(event) {
            events.push(event);
            for (const callback of listeners.get(event.type) || []) callback(event);
        },
    };
    const context = vm.createContext({
        document: {getElementById: element, createElement: () => ({})}, window, fetch,
        AbortController, AbortSignal, URLSearchParams,
        CustomEvent: class { constructor(type, options = {}) { this.type = type; this.detail = options.detail; } },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../public', filename), 'utf8'), context);
    return {context, element, events, dispatch: (type, detail) => window.dispatchEvent({type, detail})};
}

function deferred() {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return {promise, resolve};
}

const flush = () => new Promise(resolve => setImmediate(resolve));
const response = (data, ok = true) => ({ok, json: async () => data});

test('missing measured values stay unknown and do not create comparison bars', () => {
    const {context} = harness('matchup-ui.js', () => Promise.reject(new Error('Unexpected request')));
    const render = vm.runInContext('(metrics, index) => metricCell(metrics, index, "sigLandedPerMinute", number)', context);
    const known = {available: true, values: {sigLandedPerMinute: 4}, totals: {}};
    const missing = {available: false, values: {sigLandedPerMinute: 0}};
    assert.match(render([known, missing], 1), />—</);
    assert.doesNotMatch(render([known, missing], 0), /class="metric-bar"/);
    assert.doesNotMatch(render([known, missing], 1), /class="metric-bar"/);
    const zero = {available: true, values: {sigLandedPerMinute: 0}, totals: {}};
    assert.match(render([known, zero], 1), />0\.0</);
    assert.match(render([known, zero], 1), /--metric-width:0\.00%/);
});

test('fight rows preserve disputed detail and reject unsafe source URLs', () => {
    const {context} = harness('matchup-ui.js', () => Promise.reject(new Error('Unexpected request')));
    const row = vm.runInContext('fightRow', context)({
        opponent: '<img onerror=attack()>', result: 'loss', date: '2025-07-19',
        method: 'Decision', round: 3, resultUncertainty: 'Round conflicts across sources',
        url: 'javascript:attack()',
    });
    assert.match(row, /&lt;img onerror=attack\(\)&gt;/);
    assert.match(row, /Round disputed/);
    assert.match(row, /Round conflicts across sources/);
    assert.doesNotMatch(row, /href=|<img| · R3/);
});

test('superseded research is aborted and cannot replace the current error state', async () => {
    const calls = [];
    const state = harness('matchup-ui.js', (url, options) => {
        const request = deferred();
        calls.push({url, options, ...request});
        return request.promise;
    });
    const research = vm.runInContext('researchMatchup', state.context);
    const first = research({fighter1: 'First A', fighter2: 'First B', origin: 'url'});
    const second = research({fighter1: 'Second A', fighter2: 'Second B', origin: 'form'});
    assert.equal(calls[0].options.signal.aborted, true);
    assert.doesNotMatch(calls[0].url, /origin/);
    calls[1].resolve(response({error: 'Current pair unavailable'}, false));
    await second;
    calls[0].resolve(response({error: 'Stale pair unavailable'}, false));
    await first;
    assert.equal(state.element('evidence-status').textContent, 'Current pair unavailable');
    assert.equal(state.element('retry-evidence').hidden, false);
    assert.equal(state.element('evidence-content').hidden, true);
    const failures = state.events.filter(event => event.type === 'versus:research-error');
    assert.equal(failures.length, 1);
    assert.equal(failures[0].detail.fighter1, 'Second A');
});

test('a stale forecast cannot unlock a newer request or leave an old estimate visible', async () => {
    const calls = [];
    const state = harness('forecast-ui.js', (url, options) => {
        if (url === '/api/forecast-status') return Promise.resolve(response({available: true, modelVersion: 'test', supportedWeightClasses: ['Lightweight'], forecastDateRange: {max: '2099-12-31'}}));
        if (url === '/api/evaluation') return Promise.resolve(response({}));
        const request = deferred(); calls.push({url, options, ...request}); return request.promise;
    });
    await flush();
    state.dispatch('versus:evidence', {fighters: [
        {fighterId: 'a', name: 'A', latestBout: {weightClass: 'Lightweight'}},
        {fighterId: 'b', name: 'B', latestBout: {weightClass: 'Lightweight'}},
    ]});
    state.element('forecast-date').value = '2099-01-01';
    state.element('forecast-division').value = 'Lightweight';
    state.element('forecast-rounds').value = '3';
    state.element('forecast-title').value = 'no';
    const submit = state.element('forecast-form').listeners.get('submit');
    const first = submit({preventDefault() {}});
    state.element('forecast-form').listeners.get('input')();
    const second = submit({preventDefault() {}});
    assert.equal(calls[0].options.signal.aborted, true);
    calls[0].resolve(response({status: 'abstain', reasons: ['Old result']}));
    await first;
    assert.equal(state.element('forecast-submit').disabled, true);
    assert.equal(state.element('forecast-result').hidden, true);
    calls[1].resolve(response({
        status: 'forecast', fighter1: {name: 'A', probability: .7, validStatBouts: 10},
        fighter2: {name: 'B', probability: .3, validStatBouts: 8}, modelVersion: 'test',
        context: {eventDate: '2099-01-01', weightClass: 'Lightweight', scheduledRounds: 3, isTitleBout: false},
        provenance: {trainingThrough: '2022-12-31', calibrationThrough: '2024-12-31', researchDataThrough: '2026-10-31'},
    }));
    await second;
    assert.equal(state.element('forecast-submit').disabled, false);
    assert.match(state.element('forecast-result').innerHTML, /70\.0%/);
    assert.doesNotMatch(state.element('forecast-result').innerHTML, /90% confidence/);
    assert.equal(state.events.filter(event => event.type === 'versus:forecast').length, 1);
    state.dispatch('versus:research-start', {fighter1: 'C', fighter2: 'D'});
    assert.equal(state.element('forecast-result').hidden, true);
    assert.equal(state.element('forecast-panel').hidden, true);
});


test('forecast status retries after a cold-start failure and reports pending loads honestly', async () => {
    const statusRequests = [];
    let evaluationRequests = 0;
    const state = harness('forecast-ui.js', (url, options) => {
        if (url === '/api/evaluation') {
            evaluationRequests += 1;
            return Promise.resolve(response({}));
        }
        assert.equal(url, '/api/forecast-status');
        const request = deferred();
        statusRequests.push({options, ...request});
        return request.promise;
    });
    const evidence = {fighters: [
        {fighterId: 'a', name: 'A', latestBout: {weightClass: 'Lightweight'}},
        {fighterId: 'b', name: 'B', latestBout: {weightClass: 'Lightweight'}},
    ]};
    state.dispatch('versus:evidence', evidence);
    assert.equal(statusRequests.length, 1);
    assert.equal(state.element('forecast-availability').textContent, 'Loading the model…');
    assert.equal(state.element('forecast-form').hidden, true);

    statusRequests[0].resolve(response({error: 'Service is waking up'}, false));
    await flush();
    assert.match(state.element('forecast-availability').textContent, /model is unavailable/);
    assert.equal(state.element('forecast-form').hidden, true);

    state.dispatch('versus:evidence', evidence);
    assert.equal(statusRequests.length, 2);
    assert.equal(state.element('forecast-availability').textContent, 'Loading the model…');
    state.dispatch('versus:evidence', evidence);
    assert.equal(statusRequests.length, 2);
    assert.equal(evaluationRequests, 2);

    statusRequests[1].resolve(response({available: true, modelVersion: 'test', supportedWeightClasses: ['Lightweight'], forecastDateRange: {max: '2099-12-31'}}));
    await flush();
    assert.equal(state.element('forecast-form').hidden, false);
    assert.match(state.element('forecast-availability').textContent, /Last recorded division: Lightweight/);
    assert.equal(state.element('forecast-division').value, 'Lightweight');
    state.dispatch('versus:evidence', evidence);
    assert.equal(statusRequests.length, 2);
});

test('forecast diagnostics render the corrected evaluation calibration bins', async () => {
    const state = harness('forecast-ui.js', url => Promise.resolve(response(url === '/api/forecast-status'
        ? {available: true, modelVersion: 'corrected', supportedWeightClasses: ['Lightweight']}
        : {candidate: {modelVersion: 'corrected', audit: {accuracy: .64, count: 985,
            calibration_bins: [{lower: .7, upper: .8, count: 141, hit_rate: .73}]}}})));
    await flush();
    state.dispatch('versus:evidence', {fighters: [
        {fighterId: 'a', name: 'A', latestBout: {weightClass: 'Lightweight'}},
        {fighterId: 'b', name: 'B', latestBout: {weightClass: 'Lightweight'}},
    ]});
    assert.match(state.element('forecast-audit-content').innerHTML, /70–80%/);
    assert.match(state.element('forecast-audit-content').innerHTML, /<td>141<\/td>/);
    assert.match(state.element('forecast-audit-content').innerHTML, /73\.0%/);
});
