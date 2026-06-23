import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import vm from 'node:vm';
import {KEY, compareMarket, settleEntry, journalSummary, readJournal, saveJournal} from '../public/market.js';

const source = readFileSync(new URL('../public/research-ui.js', import.meta.url), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

function setup(odds = [], {statusFails = false} = {}) {
    const document = new EventTarget();
    const window = new EventTarget();
    const elements = new Map();
    class Element extends EventTarget {
        constructor(id = '') {
            super();
            this.id = id;
            this.value = '';
            this.textContent = '';
            this.dataset = {};
            this.classList = {toggle() {}};
            if (id) elements.set(id, this);
        }
        set value(value) { this.inputValue = String(value); }
        get value() { return this.inputValue; }
        set innerHTML(value) {
            this.html = value;
            if (this.id === 'journal-selection') this.value = '';
        }
        get innerHTML() { return this.html || ''; }
        querySelector(selector) { return selector === '[value="model"]' ? elements.get('model-option') : null; }
        closest(selector) {
            if (selector === '[data-use-forecast]') return this.dataset.useForecast !== undefined ? this : null;
            if (selector === '[data-quote]') return this.dataset.quote ? this : null;
            return null;
        }
        replaceChildren() {}
        scrollIntoView() {}
    }
    for (const [, id] of source.matchAll(/\$\('([^']+)'\)/g)) if (!elements.has(id)) new Element(id);
    new Element('model-option');
    document.getElementById = id => elements.get(id);
    window.matchMedia = () => ({matches:true});
    const emit = (type, detail) => window.dispatchEvent(new CustomEvent(type, {detail}));
    for (const id of ['fighter1','fighter2']) elements.get(id).addEventListener('input', () => emit('versus:matchup-change'));
    elements.get('probability-source').value = 'user';
    elements.get('odds-format').value = 'american';
    elements.get('journal-stake').value = '1';
    const values = new Map();
    const storage = {getItem:key => values.get(key) ?? null, setItem:(key,value) => values.set(key,value)};
    let nextID = 0;
    const context = vm.createContext({
        document, window, Event, CustomEvent, AbortSignal, setTimeout, Blob, URL,
        KEY, compareMarket, settleEntry, journalSummary,
        readJournal:() => readJournal(storage), saveJournal:entries => saveJournal(entries,storage),
        crypto:{randomUUID:() => `test-${++nextID}`},
        fetch:async path => ({ok: !(statusFails && path === '/api/research-status'),json:async () => path === '/api/odds'
            ? {configured:true,events:odds,fetchedAt:new Date().toISOString()}
            : {model:{ready:true},odds:{configured:true}}}),
    });
    vm.runInContext(source.replace(/^import[^\n]+\n/, ''), context);
    const element = id => elements.get(id);
    const changePair = (first, second) => {
        element('fighter1').value = first;
        element('fighter2').value = second;
        emit('versus:matchup-change');
    };
    const compare = () => {
        element('bookmaker').value = 'Test book';
        element('odds-format').value = 'american';
        element('price-f1').value = '+150';
        element('price-f2').value = '-170';
        element('quoted-at').value = '2001-01-01T12:00';
        if (element('probability-source').value === 'user') element('probability').value = '40';
        element('price-form').dispatchEvent(new Event('submit',{cancelable:true}));
        assert.equal(element('price-error').hidden, true, element('price-error').textContent);
        assert.equal(element('price-result').hidden, false);
    };
    const click = (target, receiver = document) => {
        const event = new Event('click', {cancelable:true});
        Object.defineProperty(event, 'target', {value:target});
        receiver.dispatchEvent(event);
        return event;
    };
    const useForecast = () => {
        const link = new Element();
        link.dataset.useForecast = '';
        return click(link);
    };
    const useQuote = () => {
        const button = new Element();
        button.dataset.quote = '0:0';
        click(button, element('feed-results'));
    };
    const input = id => {
        const event = new Event('input');
        Object.defineProperty(event, 'target', {value:element(id)});
        element('price-form').dispatchEvent(event);
    };
    return {element,emit,changePair,compare,useForecast,useQuote,input,journal:() => readJournal(storage)};
}

const forecast = (eventDate = '2099-01-01') => ({
    fighter1:'Alpha',fighter2:'Beta',f1Prob:.7,f2Prob:.3,
    research:{modelVersion:'test-model',dataThrough:'2022-12-31',forecastContext:{
        eventDate,weightClass:'Lightweight',rounds:5,titleBout:true,
        trainingThrough:'2022-12-31',calibrationThrough:'2024-12-31',researchDataThrough:'2026-10-31',
    }},
});

test('loaded evidence restores coverage after the initial status request fails', async () => {
    const page = setup([], {statusFails:true});
    await tick();
    assert.match(page.element('research-data-status').textContent, /could not be checked/);
    page.emit('versus:evidence', {dataThrough:'2026-10-31'});
    assert.match(page.element('research-data-status').textContent, /Oct 31, 2026/);
});

test('journal drafts follow the exact pair and survive repeated comparison and return visits', () => {
    const page = setup();
    page.changePair('Alpha','Beta');
    page.compare();
    page.element('journal-notes').value = 'Alpha has the grappling advantage.';
    page.element('journal-stake').value = '2.5';
    page.element('journal-selection').value = 'fighter1';
    page.compare();
    assert.equal(page.element('journal-notes').value, 'Alpha has the grappling advantage.');
    assert.equal(page.element('journal-stake').value, '2.5');
    assert.equal(page.element('journal-selection').value, 'fighter1');

    page.changePair('Gamma','Delta');
    assert.equal(page.element('journal-notes').value, '');
    page.compare();
    assert.equal(page.element('journal-stake').value, '1');
    assert.equal(page.element('journal-selection').value, '');
    page.element('journal-notes').value = 'Delta can control the range.';
    page.element('journal-stake').value = '0.5';
    page.element('journal-selection').value = 'fighter2';

    page.changePair('Alpha','Beta');
    page.compare();
    assert.equal(page.element('journal-notes').value, 'Alpha has the grappling advantage.');
    assert.equal(page.element('journal-stake').value, '2.5');
    assert.equal(page.element('journal-selection').value, 'fighter1');
    page.changePair('Beta','Alpha');
    page.compare();
    assert.equal(page.element('journal-notes').value, '');
    page.changePair('Gamma','Delta');
    page.compare();
    assert.equal(page.element('journal-notes').value, 'Delta can control the range.');
    assert.equal(page.element('journal-selection').value, 'fighter2');
});

test('saving a draft does not restore its saved reasoning as a new unsaved decision', () => {
    const page = setup();
    page.changePair('Alpha','Beta');
    page.compare();
    page.element('journal-notes').value = 'Saved reasoning';
    page.element('journal-selection').value = 'fighter1';
    page.element('save-research').dispatchEvent(new Event('submit',{cancelable:true}));
    assert.equal(page.journal()[0].notes, 'Saved reasoning');
    page.changePair('Gamma','Delta');
    page.compare();
    page.changePair('Alpha','Beta');
    page.compare();
    assert.equal(page.element('journal-notes').value, '');
    assert.equal(page.element('journal-selection').value, '');
});

test('forecast handoff is explicit, preserves normal anchor navigation, and shows its full setup', () => {
    const page = setup();
    page.changePair('Alpha','Beta');
    page.element('probability').value = '42';
    page.emit('versus:forecast', forecast());
    assert.equal(page.element('probability-source').value, 'user');
    assert.equal(page.element('probability').value, '42');
    assert.equal(page.useForecast().defaultPrevented, false);
    assert.equal(page.element('probability-source').value, 'model');
    assert.equal(page.element('probability').value, '70.00');
    assert.equal(page.element('probability').readOnly, true);
    assert.match(page.element('estimate-context').textContent, /Jan 1, 2099 · Lightweight · 5 rounds · title bout/);
    page.compare();
    assert.match(page.element('market-rows').innerHTML, /role="rowheader"/);
    assert.match(page.element('market-rows').innerHTML, /role="cell" data-label="Model estimate">70\.0%/);
    assert.match(page.element('market-rows').innerHTML, /data-label="Expected return per unit"/);
    page.emit('versus:forecast-invalidated');
    page.element('probability').value = '42';
    page.useForecast();
    assert.equal(page.element('probability-source').value, 'user');
    assert.equal(page.element('probability').value, '42');
});

test('quoted event date warning uses UTC consistently and clears when a quote becomes manual', async () => {
    const fight = {
        homeTeam:'Alpha',awayTeam:'Beta',startTime:'2099-01-02T00:30:00+02:00',
        bookmakers:[{title:'Provider book',lastUpdate:'2001-01-01T12:00:00Z',outcomes:[{name:'Alpha',price:2.5},{name:'Beta',price:1.6}]}],
    };
    const page = setup([fight]);
    page.element('load-prices').dispatchEvent(new Event('click'));
    await tick();
    page.useQuote();
    page.emit('versus:forecast', forecast());
    page.useForecast();
    assert.doesNotMatch(page.element('estimate-context').textContent, /differs from/);
    fight.startTime = '2099-01-03T00:30:00+02:00';
    page.useQuote();
    page.emit('versus:forecast', forecast());
    page.useForecast();
    assert.match(page.element('estimate-context').textContent, /UTC date is Jan 2, 2099, which differs from the estimate date/);
    page.input('bookmaker');
    assert.doesNotMatch(page.element('estimate-context').textContent, /differs from/);
    assert.equal(page.element('probability').value, '70.00');
});
