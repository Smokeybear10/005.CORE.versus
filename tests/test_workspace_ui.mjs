import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import vm from 'node:vm';
import {compareMarket, journalSummary} from '../public/market.js';

const workspaceSource = readFileSync(new URL('../public/workspace-ui.js', import.meta.url), 'utf8');
const pickerSource = readFileSync(new URL('../public/script.js', import.meta.url), 'utf8');
const pricesSource = readFileSync(new URL('../public/research-ui.js', import.meta.url), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

function setup(url = 'http://localhost/predict.html', {picker = false, clipboardFails = false, prices = false, odds = []} = {}) {
    const document = new EventTarget();
    const elements = new Map();
    class Element extends EventTarget {
        constructor(id = '', tagName = 'DIV') {
            super();
            this.id = id;
            this.tagName = tagName;
            this.dataset = {};
            this.attributes = new Map();
            this.children = [];
            this.value = '';
            this.hidden = false;
            this.textContent = '';
            this.classList = {add() {}, remove() {}, toggle() {}};
            if (id) elements.set(id, this);
        }
        setAttribute(key, value) { this.attributes.set(key, value); }
        set value(value) { this.inputValue = String(value); }
        get value() { return this.inputValue; }
        getAttribute(key) { return this.attributes.get(key); }
        removeAttribute(key) { this.attributes.delete(key); }
        focus() { document.activeElement = this; }
        select() { this.selected = true; }
        replaceChildren() { this.children = []; }
        appendChild(child) { child.parentElement = this; this.children.push(child); }
        querySelector(selector) {
            if (selector === '[value="model"]') return elements.get('model-option');
            return null;
        }
        closest(selector) {
            if (selector === '[data-quote]') return this.dataset.quote ? this : null;
            for (let node = this; node; node = node.parentElement) if (node.dataset.workspacePanel) return node;
            return null;
        }
        getBoundingClientRect() { return this.rect || {top:0,bottom:0}; }
        scrollIntoView(options) { this.scrolled = true; this.scrollOptions = options; this.scrollCalls = (this.scrollCalls || 0) + 1; }
        requestSubmit() { this.dispatchEvent(new Event('submit', {cancelable:true})); }
    }
    const views = ['report', 'estimate', 'prices', 'journal'];
    const tabs = views.map(view => {
        const element = new Element(`tab-${view}`, 'BUTTON');
        element.dataset.workspaceTab = view;
        return element;
    });
    const panels = views.map(view => {
        const element = new Element(`view-${view}`);
        element.dataset.workspacePanel = view;
        return element;
    });
    for (const id of ['workspace-navigation','workspace-empty','workspace-empty-title','workspace-empty-description','workspace-selection','share-matchup','share-status','share-link','matchup-form','fighter1','fighter2','autocomplete1','autocomplete2','predict-btn','form-error','form-status','retry-fighters','result-section','prediction-top','example-matchup']) new Element(id);
    if (prices) {
        for (const [, id] of pricesSource.matchAll(/\$\('([^']+)'\)/g)) if (!elements.has(id)) new Element(id);
        new Element('model-option');
        elements.get('probability-source').value = 'user';
        elements.get('odds-format').value = 'american';
    }
    const career = new Element('career-history-section', 'DETAILS');
    panels[0].appendChild(career);
    document.getElementById = id => elements.get(id) || null;
    document.querySelectorAll = selector => selector === '[data-workspace-tab]' ? tabs : selector === '[data-workspace-panel]' ? panels : [];
    document.createElement = tag => new Element('', tag.toUpperCase());
    const window = new EventTarget();
    window.location = new URL(url);
    window.matchMedia = () => ({matches:true});
    const history = [];
    const historyEntries = [window.location.href];
    let historyIndex = 0;
    window.history = Object.fromEntries(['push','replace'].map(mode => [`${mode}State`, (_state, _unused, next) => {
        window.location = new URL(next, window.location);
        history.push({mode, url:window.location.href});
        if (mode === 'push') {
            historyEntries.splice(historyIndex + 1);
            historyIndex += 1;
        }
        historyEntries[historyIndex] = window.location.href;
    }]));
    window.history.back = () => {
        if (historyIndex === 0) return;
        window.location = new URL(historyEntries[--historyIndex]);
        window.dispatchEvent(new Event('popstate'));
    };
    const copied = [];
    const navigator = {clipboard:{writeText:async value => {
        if (clipboardFails) throw new Error('Clipboard denied');
        copied.push(value);
    }}};
    let deliverFighters;
    const fighterResponse = new Promise(resolve => { deliverFighters = resolve; });
    const fetch = path => path === '/api/fighters' ? fighterResponse : Promise.resolve({ok:true,json:async () => path === '/api/odds'
        ? {configured:true,events:odds,fetchedAt:new Date().toISOString()}
        : {models:{},model:{ready:true},odds:{configured:true}}});
    let journal = [];
    const context = vm.createContext({document,window,navigator,URL,URLSearchParams,Event,CustomEvent,AbortSignal,fetch,setTimeout,clearTimeout,
        KEY:'test-journal',compareMarket,journalSummary,readJournal:() => journal,saveJournal:entries => { journal = entries; return journal; },
        crypto:{randomUUID:() => `entry-${journal.length}`}});
    vm.runInContext(workspaceSource, context);
    if (prices) vm.runInContext(pricesSource.replace(/^import[^\n]+\n/, ''), context);
    if (picker) {
        vm.runInContext(pickerSource, context);
        document.dispatchEvent(new Event('DOMContentLoaded'));
    }
    const emit = (name, detail) => window.dispatchEvent(new CustomEvent(name, {detail}));
    const start = (fighter1 = 'Alpha', fighter2 = 'Beta', origin = 'form') => emit('versus:research-start', {fighter1,fighter2,origin});
    return {document,window,navigator,elements,tabs,panels,career,history,copied,emit,start,
        journal:() => journal,
        useQuote:() => {
            const button = new Element('', 'BUTTON');
            button.dataset.quote = '0:0';
            const event = new Event('click');
            Object.defineProperty(event, 'target', {value:button});
            elements.get('feed-results').dispatchEvent(event);
        },
        loaded:names => deliverFighters({ok:true,json:async () => ({fighters:names})})};
}

test('shared matchup waits for fighter lookup, canonicalizes names and preserves its view', async () => {
    const page = setup('http://localhost/predict.html?fighter1=alpha&fighter2=bEtA&view=estimate', {picker:true});
    const starts = [];
    page.window.addEventListener('versus:research-start', event => starts.push(event.detail));
    assert.equal(starts.length, 0);
    assert.equal(page.elements.get('predict-btn').disabled, true);
    page.loaded(['Alpha','Beta']);
    await tick();
    assert.equal(starts.length, 1);
    assert.equal(starts[0].fighter1, 'Alpha');
    assert.equal(starts[0].fighter2, 'Beta');
    assert.equal(starts[0].origin, 'url');
    assert.equal(page.tabs[1].getAttribute('aria-selected'), 'true');
    assert.equal(page.window.location.searchParams.get('fighter1'), 'Alpha');
    assert.equal(page.elements.get('workspace-empty-title').textContent, 'Loading the matchup');
    page.emit('versus:evidence', {fighters:[{name:'Alpha'},{name:'Beta'}]});
    assert.equal(page.elements.get('workspace-empty').hidden, true);
});

test('unknown shared fighters never start research or enable sharing', async () => {
    const page = setup('http://localhost/predict.html?fighter1=Unknown&fighter2=Beta', {picker:true});
    const starts = [];
    page.window.addEventListener('versus:research-start', event => starts.push(event.detail));
    page.loaded(['Alpha','Beta']);
    await tick();
    assert.equal(starts.length, 0);
    assert.equal(page.elements.get('share-matchup').disabled, true);
    assert.match(page.elements.get('form-error').textContent, /Choose a fighter/);
});

test('tab keys move focus and active panel without intercepting normal Tab', () => {
    const page = setup();
    const press = (element, key) => {
        const event = new Event('keydown', {cancelable:true});
        Object.defineProperty(event, 'key', {value:key});
        element.dispatchEvent(event);
        return event;
    };
    assert.equal(press(page.tabs[0], 'ArrowRight').defaultPrevented, true);
    assert.equal(page.document.activeElement, page.tabs[1]);
    assert.equal(page.panels[1].hidden, false);
    assert.equal(page.tabs.filter(tab => tab.tabIndex === 0).length, 1);
    press(page.tabs[1], 'End');
    assert.equal(page.panels[3].hidden, false);
    press(page.tabs[3], 'Home');
    assert.equal(page.panels[0].hidden, false);
    assert.equal(press(page.tabs[0], 'Tab').defaultPrevented, false);
});

test('changing tabs from a deep report reveals the beginning of the new view', () => {
    const page = setup();
    page.elements.get('workspace-navigation').rect = {top:60,bottom:112};
    page.panels[0].rect = {top:-900,bottom:2400};
    page.tabs[2].dispatchEvent(new Event('click'));
    assert.equal(page.panels[2].scrolled, true);
    assert.equal(page.panels[2].scrollOptions.block, 'start');
    assert.equal(page.panels[2].hidden, false);
    page.tabs[2].dispatchEvent(new Event('click'));
    assert.equal(page.panels[2].scrollCalls, 1);
});

test('changing tabs near the picker does not move the page', () => {
    const page = setup();
    page.elements.get('workspace-navigation').rect = {top:290,bottom:342};
    page.panels[0].rect = {top:342,bottom:2400};
    page.tabs[1].dispatchEvent(new Event('click'));
    assert.equal(page.panels[1].scrolled, undefined);
});

test('keyboard fighter selection keeps the active suggestion visible', async () => {
    const page = setup(undefined, {picker:true});
    const names = Array.from({length:10}, (_, index) => `Fighter ${index}`);
    page.loaded(names);
    await tick();
    const input = page.elements.get('fighter1');
    input.value = 'Fighter';
    input.dispatchEvent(new Event('input'));
    const press = key => {
        const event = new Event('keydown', {cancelable:true});
        Object.defineProperty(event, 'key', {value:key});
        input.dispatchEvent(event);
    };
    for (let index = 0; index < 10; index++) press('ArrowDown');
    const options = page.elements.get('autocomplete1').children;
    assert.equal(input.getAttribute('aria-activedescendant'), options[9].id);
    assert.equal(options[9].scrollOptions.block, 'nearest');
    assert.equal(options[9].getAttribute('aria-selected'), 'true');
    press('Enter');
    assert.equal(input.value, 'Fighter 9');
    assert.equal(input.getAttribute('aria-expanded'), 'false');
});

test('fighter lookup ranks name and word prefixes before substring matches and the result limit', async () => {
    const page = setup(undefined, {picker:true});
    page.loaded([
        ...Array.from({length:12}, (_, index) => `Antonio${index} Jones`),
        'B Jonsson', 'Carls John De Tomas', 'Jon Jones', 'Jonathan Martinez', 'Tom Aspinall', 'Aptom Jones',
    ]);
    await tick();
    const input = page.elements.get('fighter1');
    input.value = 'Jon';
    input.dispatchEvent(new Event('input'));
    const options = page.elements.get('autocomplete1').children;
    assert.equal(options.length, 10);
    assert.deepEqual(options.slice(0, 3).map(item => item.textContent), ['Jon Jones','Jonathan Martinez','Antonio0 Jones']);
    input.value = 'Tom';
    input.dispatchEvent(new Event('input'));
    assert.deepEqual(page.elements.get('autocomplete1').children.map(item => item.textContent), ['Tom Aspinall','Carls John De Tomas','Aptom Jones']);
});

test('fighter search and exact validation normalize accents and repeated spaces', async () => {
    const page = setup(undefined, {picker:true});
    const starts = [];
    page.window.addEventListener('versus:research-start', event => starts.push(event.detail));
    page.loaded(['José Aldo','Tom Aspinall']);
    await tick();
    const first = page.elements.get('fighter1');
    first.value = '  JOSE   ALDO  ';
    first.dispatchEvent(new Event('input'));
    assert.equal(page.elements.get('autocomplete1').children[0]?.textContent, 'José Aldo');
    page.elements.get('fighter2').value = '  tom   aspinall ';
    page.elements.get('matchup-form').requestSubmit();
    assert.equal(starts.length, 1);
    assert.equal(starts[0].fighter1, 'José Aldo');
    assert.equal(starts[0].fighter2, 'Tom Aspinall');
});

test('empty fighter search results explain the miss without becoming selectable', async () => {
    const page = setup(undefined, {picker:true});
    page.loaded(['Jon Jones','Tom Aspinall']);
    await tick();
    const input = page.elements.get('fighter1');
    input.value = 'Nobody Here';
    input.dispatchEvent(new Event('input'));
    const children = page.elements.get('autocomplete1').children;
    assert.equal(children.length, 1);
    assert.match(children[0].textContent, /No matching fighters/);
    assert.notEqual(children[0].getAttribute('role'), 'option');
    for (const key of ['ArrowDown','Enter']) {
        const event = new Event('keydown', {cancelable:true});
        Object.defineProperty(event, 'key', {value:key});
        input.dispatchEvent(event);
    }
    assert.equal(input.value, 'Nobody Here');
    assert.equal(input.getAttribute('aria-activedescendant'), undefined);
});

test('focusing a selected fighter reopens suggestions without invalidating the matchup', async () => {
    const page = setup('http://localhost/research?fighter1=Jon+Jones&fighter2=Tom+Aspinall', {picker:true});
    page.loaded(['Jon Jones','Tom Aspinall']);
    await tick();
    page.emit('versus:evidence', {fighters:[{name:'Jon Jones'},{name:'Tom Aspinall'}]});
    let changes = 0;
    page.window.addEventListener('versus:matchup-change', () => changes++);
    const input = page.elements.get('fighter1');
    input.dispatchEvent(new Event('blur'));
    input.dispatchEvent(new Event('focus'));
    assert.equal(input.getAttribute('aria-expanded'), 'true');
    assert.equal(page.elements.get('autocomplete1').children[0]?.textContent, 'Jon Jones');
    assert.equal(changes, 0);
    assert.equal(page.elements.get('share-matchup').disabled, false);
    assert.equal(page.elements.get('workspace-empty').hidden, true);
    assert.equal(page.window.location.searchParams.get('fighter1'), 'Jon Jones');
});

test('returning to an empty URL clears stale picker errors and invalid states', async () => {
    const page = setup('http://localhost/predict.html?fighter1=Alpha&fighter2=Alpha', {picker:true});
    page.loaded(['Alpha','Beta']);
    await tick();
    assert.match(page.elements.get('form-error').textContent, /two different/);
    assert.equal(page.elements.get('fighter2').getAttribute('aria-invalid'), 'true');
    page.window.location = new URL('http://localhost/predict.html?view=report');
    page.window.dispatchEvent(new Event('popstate'));
    assert.equal(page.elements.get('form-error').hidden, true);
    assert.equal(page.elements.get('fighter2').getAttribute('aria-invalid'), undefined);
    assert.equal(page.elements.get('fighter1').value, '');
    assert.equal(page.elements.get('fighter2').value, '');
});

test('stale evidence and errors do not affect a replacement matchup', () => {
    const page = setup();
    page.start('Alpha','Beta');
    page.start('Alpha','Gamma');
    page.tabs[1].dispatchEvent(new Event('click'));
    page.emit('versus:evidence', {fighters:[{name:'Alpha'},{name:'Beta'}]});
    assert.equal(page.elements.get('workspace-empty').hidden, false);
    page.emit('versus:research-error', {fighter1:'Alpha',fighter2:'Beta'});
    assert.equal(page.elements.get('workspace-empty-title').textContent, 'Loading the matchup');
    page.emit('versus:research-error', {fighter1:'Alpha',fighter2:'Gamma'});
    assert.equal(page.elements.get('workspace-empty-title').textContent, 'The report needs to load first');
    page.emit('versus:research-loading', {fighter1:'Alpha',fighter2:'Gamma'});
    assert.equal(page.elements.get('workspace-empty-title').textContent, 'Loading the matchup');
    page.emit('versus:evidence', {fighters:[{name:'Alpha'},{name:'Gamma'}]});
    assert.equal(page.elements.get('workspace-empty').hidden, true);
});

test('editing a fighter invalidates share state and removes the previous pair from the URL', () => {
    const page = setup();
    page.start();
    page.emit('versus:matchup-change');
    assert.equal(page.window.location.searchParams.has('fighter1'), false);
    assert.equal(page.elements.get('share-matchup').disabled, true);
    assert.equal(page.elements.get('workspace-empty').hidden, false);
    page.emit('versus:evidence', {fighters:[{name:'Alpha'},{name:'Beta'}]});
    assert.equal(page.elements.get('workspace-empty').hidden, false);
});

test('a deep history link reveals the report and opens its collapsed section', () => {
    const page = setup();
    page.start();
    page.tabs[2].dispatchEvent(new Event('click'));
    page.window.location.hash = '#career-history-section';
    page.window.dispatchEvent(new Event('hashchange'));
    assert.equal(page.panels[0].hidden, false);
    assert.equal(page.career.open, true);
    assert.equal(page.career.scrolled, true);
    assert.equal(page.window.location.searchParams.get('view'), 'report');
});

test('browser history restores another pair through the validated picker', async () => {
    const page = setup('http://localhost/predict.html?fighter1=Alpha&fighter2=Beta&view=report', {picker:true});
    page.loaded(['Alpha','Beta','Gamma']);
    await tick();
    page.window.location = new URL('http://localhost/predict.html?fighter1=Alpha&fighter2=Gamma&view=prices');
    page.window.dispatchEvent(new Event('popstate'));
    assert.equal(page.elements.get('fighter2').value, 'Gamma');
    assert.equal(page.elements.get('workspace-selection').textContent, 'Alpha vs Gamma');
    assert.equal(page.panels[2].hidden, false);
    page.window.location = new URL('http://localhost/predict.html?view=journal');
    page.window.dispatchEvent(new Event('popstate'));
    assert.equal(page.elements.get('fighter1').value, '');
    assert.equal(page.elements.get('share-matchup').disabled, true);
    assert.equal(page.panels[3].hidden, false);
});

test('editing and comparing another fighter preserves the previous matchup in Back history', async () => {
    const page = setup('http://localhost/predict.html?fighter1=Alpha&fighter2=Beta&view=report', {picker:true});
    page.loaded(['Alpha','Beta','Gamma']);
    await tick();
    const input = page.elements.get('fighter2');
    input.value = 'Ga';
    input.dispatchEvent(new Event('input'));
    input.value = 'Gamma';
    input.dispatchEvent(new Event('input'));
    assert.equal(page.window.location.searchParams.has('fighter1'), false);
    page.elements.get('matchup-form').requestSubmit();
    assert.equal(page.window.location.searchParams.get('fighter2'), 'Gamma');
    page.window.history.back();
    assert.equal(page.window.location.searchParams.get('fighter2'), 'Beta');
    assert.equal(input.value, 'Beta');
    assert.equal(page.elements.get('workspace-selection').textContent, 'Alpha vs Beta');
    assert.equal(page.elements.get('share-matchup').disabled, false);
});

test('Back during an incomplete edit restores the last valid matchup', async () => {
    const page = setup('http://localhost/predict.html?fighter1=Alpha&fighter2=Beta&view=report', {picker:true});
    page.loaded(['Alpha','Beta','Gamma']);
    await tick();
    page.elements.get('fighter2').value = 'G';
    page.elements.get('fighter2').dispatchEvent(new Event('input'));
    page.tabs[2].dispatchEvent(new Event('click'));
    page.window.history.back();
    assert.equal(page.elements.get('fighter2').value, 'Beta');
    assert.equal(page.panels[0].hidden, false);
    assert.equal(page.elements.get('workspace-selection').textContent, 'Alpha vs Beta');
});

test('sharing contains only the canonical matchup and selected view', async () => {
    const page = setup('http://localhost/predict.html?notes=private&token=secret#journal');
    page.start();
    page.tabs[2].dispatchEvent(new Event('click'));
    page.elements.get('share-matchup').dispatchEvent(new Event('click'));
    await tick();
    const url = new URL(page.copied[0]);
    assert.deepEqual([...url.searchParams.keys()], ['fighter1','fighter2','view']);
    assert.equal(url.searchParams.get('view'), 'prices');
    assert.equal(url.hash, '');
    assert.equal(page.elements.get('share-status').textContent, 'Matchup link copied.');
});

test('clipboard denial exposes a selected read-only link without claiming success', async () => {
    const page = setup(undefined, {clipboardFails:true});
    page.start();
    page.elements.get('share-matchup').dispatchEvent(new Event('click'));
    await tick();
    assert.equal(page.elements.get('share-link').hidden, false);
    assert.equal(page.elements.get('share-link').selected, true);
    assert.match(page.elements.get('share-link').value, /fighter1=Alpha&fighter2=Beta/);
    assert.equal(page.elements.get('share-status').textContent, 'Copy the selected link.');
});

test('a delayed clipboard failure does not expose the previous matchup link after editing', async () => {
    const page = setup();
    let rejectCopy;
    page.navigator.clipboard.writeText = () => new Promise((_resolve, reject) => { rejectCopy = reject; });
    page.start();
    page.elements.get('share-matchup').dispatchEvent(new Event('click'));
    page.emit('versus:matchup-change');
    rejectCopy(new Error('Clipboard denied'));
    await tick();
    assert.equal(page.elements.get('share-link').hidden, true);
    assert.equal(page.elements.get('share-status').textContent, '');
});

test('comparing an unchanged matchup preserves manual prices and probability', async () => {
    const page = setup('http://localhost/predict.html?fighter1=Alpha&fighter2=Beta', {picker:true,prices:true});
    page.loaded(['Alpha','Beta']);
    await tick();
    page.elements.get('bookmaker').value = 'Manual book';
    page.elements.get('price-f1').value = '-120';
    page.elements.get('price-f2').value = '+110';
    page.elements.get('probability').value = '58';
    const quotedAt = page.elements.get('quoted-at').value;
    page.elements.get('matchup-form').requestSubmit();
    assert.equal(page.elements.get('price-f1').value, '-120');
    assert.equal(page.elements.get('price-f2').value, '+110');
    assert.equal(page.elements.get('probability').value, '58');
    assert.equal(page.elements.get('quoted-at').value, quotedAt);
    page.elements.get('price-form').requestSubmit();
    assert.equal(page.elements.get('price-error').hidden, true);
    assert.match(page.elements.get('market-caption').textContent, /user-entered quote/);
});

test('provider quote provenance survives compare and clears when a fighter changes', async () => {
    const lastUpdate = new Date(Date.now() - 3600000).toISOString();
    const startTime = new Date(Date.now() + 86400000).toISOString();
    const odds = [{homeTeam:'Alpha',awayTeam:'Beta',startTime,bookmakers:[{title:'Source book',lastUpdate,outcomes:[{name:'Alpha',price:2.2},{name:'Beta',price:1.8}]}]}];
    const page = setup('http://localhost/predict.html?fighter1=Alpha&fighter2=Beta', {picker:true,prices:true,odds});
    page.loaded(['Alpha','Beta','Gamma']);
    await tick();
    page.elements.get('load-prices').dispatchEvent(new Event('click'));
    await tick();
    page.useQuote();
    page.elements.get('probability').value = '55';
    page.elements.get('matchup-form').requestSubmit();
    assert.equal(page.elements.get('price-f1').value, '2.2');
    assert.equal(page.elements.get('price-f2').value, '1.8');
    page.elements.get('price-form').requestSubmit();
    assert.equal(page.elements.get('price-error').hidden, true);
    assert.match(page.elements.get('market-caption').textContent, /The Odds API/);
    page.elements.get('journal-selection').value = 'fighter1';
    page.elements.get('journal-stake').value = '1';
    page.elements.get('save-research').requestSubmit();
    assert.equal(page.journal()[0].quoteSource, 'the_odds_api');
    assert.equal(page.journal()[0].quotedAt, lastUpdate);
    assert.equal(page.journal()[0].eventStartTime, startTime);
    page.elements.get('fighter1').value = 'Gamma';
    page.elements.get('fighter1').dispatchEvent(new Event('input'));
    assert.equal(page.elements.get('price-f1').value, '');
    assert.equal(page.elements.get('price-f2').value, '');
    assert.equal(page.elements.get('probability').value, '');
    page.elements.get('odds-format').value = 'american';
    page.elements.get('price-f1').value = '-110';
    page.elements.get('price-f2').value = '-110';
    page.elements.get('probability').value = '57';
    page.elements.get('price-form').requestSubmit();
    page.elements.get('journal-selection').value = 'fighter1';
    page.elements.get('save-research').requestSubmit();
    assert.equal(page.journal()[0].fighter1, 'Gamma');
    assert.equal(page.journal()[0].quoteSource, 'manual');
    assert.equal(page.journal()[0].eventStartTime, null);
});
