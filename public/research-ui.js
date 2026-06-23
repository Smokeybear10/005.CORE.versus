import { KEY, compareMarket, settleEntry, journalSummary, readJournal, saveJournal } from './market.js';

const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const percent = value => `${(value * 100).toFixed(1)}%`;
const signed = (value, digits = 2) => `${value > 0 ? '+' : ''}${value.toFixed(digits)}`;
const american = value => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(0)}`;
const dateTime = value => new Date(value).toLocaleString(undefined, {month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit'});
const day = value => value ? new Date(`${value.slice(0,10)}T12:00:00Z`).toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}) : 'not recorded';
const localTime = value => {
    const date = new Date(value);
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0,16);
};
let prediction = null;
let readiness = null;
let evidenceThrough = null;
let comparison = null;
let quoteSource = 'manual';
let eventStartTime = null;
let providerQuotedAt = null;
let entries = [];
let storageError = false;
let lastRemoved = null;
let feedEvents = [];
const journalDrafts = new Map();
let draftPair = null;

async function getJSON(path) {
    const response = await fetch(path, {signal:AbortSignal.timeout(path === "/api/odds" ? 15000 : 55000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The request could not be completed.');
    return result;
}
function message(element, text, error = false) {
    element.textContent = text;
    element.hidden = !text;
    element.classList.toggle('research-error', error);
}
function names() { return [$('fighter1').value.trim(), $('fighter2').value.trim()]; }
function syncLabels() {
    const [first, second] = names();
    $('quote-f1-name').textContent = first || 'First fighter';
    $('quote-f2-name').textContent = second || 'Second fighter';
    $('estimate-label').textContent = `${first || 'First fighter'}’s win probability`;
}
function invalidatePrice() {
    comparison = null;
    $('price-result').hidden = true;
    message($('price-error'), '');
    message($('save-status'), '');
}
function rememberJournalDraft() {
    if (!draftPair) return;
    journalDrafts.set(draftPair, {
        notes:$('journal-notes').value, stake:$('journal-stake').value, selection:$('journal-selection').value,
    });
}
function restoreJournalDraft(fighter1, fighter2) {
    draftPair = JSON.stringify([fighter1, fighter2]);
    const draft = journalDrafts.get(draftPair);
    $('journal-notes').value = draft?.notes ?? '';
    $('journal-stake').value = draft?.stake ?? '1';
    $('journal-selection').value = draft?.selection ?? '';
}
function renderEstimateContext() {
    const useModel = $('probability-source').value === 'model' && prediction;
    if (!useModel) {
        $('estimate-context').classList.toggle('forecast-mismatch', false);
        $('estimate-context').textContent = 'Use your own probability estimate. A high win probability can still be a poor price.';
        return;
    }
    const context = prediction.research?.forecastContext;
    const setup = context
        ? `${day(context.eventDate)} · ${context.weightClass} · ${context.rounds} rounds · ${context.titleBout ? 'title bout' : 'non-title bout'}`
        : 'Model estimate for this matchup';
    const quotedEventDate = eventStartTime ? new Date(eventStartTime).toISOString().slice(0,10) : null;
    const mismatch = context && quotedEventDate && context.eventDate !== quotedEventDate;
    $('estimate-context').classList.toggle('forecast-mismatch', Boolean(mismatch));
    $('estimate-context').textContent = `${setup}. Recorded data through ${day(context?.researchDataThrough || prediction.research?.dataThrough)}. An advantage over market prices has not been established.${mismatch ? ` The quoted event's UTC date is ${day(quotedEventDate)}, which differs from the estimate date. Confirm the fight setup before using this quote.` : ''}`;
}
function updateEstimate() {
    invalidatePrice();
    const useModel = $('probability-source').value === 'model' && prediction;
    $('probability').readOnly = Boolean(useModel);
    $('probability').value = useModel ? (prediction.f1Prob * 100).toFixed(2) : '';
    renderEstimateContext();
}

$('quoted-at').value = localTime(new Date());
syncLabels();
window.addEventListener('versus:evidence', event => {
    evidenceThrough = event.detail.dataThrough;
    if (evidenceThrough) $('research-data-status').textContent = `Recorded fights through ${day(evidenceThrough)}`;
});
getJSON('/api/research-status').then(data => {
    readiness = data;
    $('research-data-status').textContent = data.research?.available || data.model.ready
        ? `Recorded fights through ${day(data.research?.dataThrough || data.model.dataThrough)}`
        : 'Recorded fight data is unavailable';
    $('load-prices').hidden = !data.odds.configured;
    $('quote-source-status').textContent = data.odds.configured
        ? 'Enter a quote or load upcoming moneylines from The Odds API. Compare both sides from one bookmaker.'
        : 'Enter both moneylines from the same bookmaker. No live price feed is connected.';
}).catch(() => {
    if (!evidenceThrough) $('research-data-status').textContent = 'Model coverage could not be checked · manual price comparison is available';
});

window.addEventListener('versus:matchup-change', () => {
    rememberJournalDraft();
    draftPair = null;
    $('journal-notes').value = '';
    $('journal-stake').value = '1';
    $('journal-selection').value = '';
    prediction = null;
    $('model-panel').hidden = true;
    $('model-details').open = false;
    const option = $('probability-source').querySelector('[value="model"]');
    option.disabled = true;
    option.textContent = 'Model estimate · set up a forecast first';
    $('probability-source').value = 'user';
    $('price-f1').value = '';
    $('price-f2').value = '';
    quoteSource = 'manual';
    eventStartTime = null;
    providerQuotedAt = null;
    syncLabels();
    updateEstimate();
});
window.addEventListener('versus:forecast', event => {
    const [first, second] = names();
    if (event.detail.fighter1 !== first || event.detail.fighter2 !== second) return;
    prediction = event.detail;
    syncLabels();
    const option = $('probability-source').querySelector('[value="model"]');
    option.disabled = false;
    option.textContent = 'Model estimate';
    invalidatePrice();
    if ($('probability-source').value === 'model') updateEstimate();

});
window.addEventListener('versus:forecast-invalidated', () => {
    prediction = null;
    $('probability-source').querySelector('[value="model"]').disabled = true;
    if ($('probability-source').value === 'model') {
        $('probability-source').value = 'user';
        updateEstimate();
    }
});
$('probability-source').addEventListener('change', updateEstimate);
document.addEventListener('click', event => {
    if (!event.target.closest?.('[data-use-forecast]')) return;
    const [first, second] = names();
    if (!prediction || prediction.fighter1 !== first || prediction.fighter2 !== second) return;
    $('probability-source').value = 'model';
    updateEstimate();
});
$('odds-format').addEventListener('change', () => {
    $('price-f1').value = '';
    $('price-f2').value = '';
    const isAmerican = $('odds-format').value === 'american';
    const placeholder = isAmerican ? 'e.g. -110' : 'e.g. 1.91';
    $('price-f1').inputMode = isAmerican ? 'text' : 'decimal';
    $('price-f2').inputMode = isAmerican ? 'text' : 'decimal';
    $('price-f1').placeholder = placeholder;
    $('price-f2').placeholder = placeholder;
});
$('price-form').addEventListener('input', event => {
    invalidatePrice();
    if (['bookmaker','price-f1','price-f2','quoted-at','odds-format'].includes(event.target.id)) {
        quoteSource = 'manual';
        eventStartTime = null;
        providerQuotedAt = null;
        renderEstimateContext();
    }
});
$('price-form').addEventListener('change', invalidatePrice);
$('price-form').addEventListener('submit', event => {
    event.preventDefault();
    invalidatePrice();
    try {
        const [fighter1, fighter2] = names();
        if (!fighter1 || !fighter2 || fighter1.toLowerCase() === fighter2.toLowerCase()) throw new Error('Enter two different fighters in the matchup above.');
        const bookmaker = $('bookmaker').value.trim();
        if (!bookmaker) throw new Error('Enter the bookmaker so this quote has a source.');
        const rawProbability = $('probability').value.trim();
        if (!rawProbability) throw new Error('Enter your win-probability estimate or choose the research model after estimating the matchup.');
        const source = $('probability-source').value;
        if (source === 'model' && !prediction) throw new Error('Estimate this matchup before using its model probability.');
        const probability1 = source === 'model' ? prediction.f1Prob : Number(rawProbability) / 100;
        if (!Number.isFinite(probability1) || probability1 < 0 || probability1 > 1) throw new Error('Win probability must be between 0% and 100%.');
        const quoteDate = new Date(providerQuotedAt || $('quoted-at').value);
        if (!Number.isFinite(quoteDate.getTime()) || quoteDate.getTime() > Date.now()) throw new Error('Enter when you observed the quote. It cannot be in the future.');
        const result = compareMarket({odds1:$('price-f1').value.replaceAll('−','-'),odds2:$('price-f2').value.replaceAll('−','-'),format:$('odds-format').value,probability1});
        comparison = {fighter1,fighter2,bookmaker,quotedAt:quoteDate.toISOString(),probabilitySource:source,result,quoteSource,eventStartTime,modelContext:source === 'model' ? prediction.research : null};
        $('probability-column').textContent = source === 'model' ? 'Model estimate' : 'Your estimate';
        $('market-caption').textContent = `${bookmaker} · ${dateTime(comparison.quotedAt)} · ${quoteSource === 'manual' ? 'user-entered quote' : 'The Odds API'}`;
        $('market-rows').innerHTML = result.sides.map((side,index) => `<tr role="row"><th scope="row" role="rowheader">${escape(index === 0 ? fighter1 : fighter2)}</th><td role="cell" data-label="Offered price">${american(side.american)}<small>${side.decimal.toFixed(2)} decimal</small></td><td role="cell" data-label="Break-even">${percent(side.breakEven)}</td><td role="cell" data-label="Market without margin">${percent(side.fairProbability)}</td><td role="cell" data-label="${source === 'model' ? 'Model estimate' : 'Your estimate'}">${percent(side.modelProbability)}</td><td role="cell" data-label="Gap">${signed(side.gap * 100,1)} pp</td><td role="cell" data-label="Expected return per unit">${signed(side.expectedReturn,3)} u<small>${signed(side.expectedReturn * 100,1)}%</small></td></tr>`).join('');
        const ageMinutes = Math.floor((Date.now() - quoteDate.getTime()) / 60000);
        $('market-context').textContent = `Bookmaker overround: ${signed(result.overround * 100,2)}%. ${result.overround < 0 ? 'These prices imply less than 100% together; confirm they refer to the same market and time. ' : ''}${ageMinutes >= 60 ? 'This quote is over an hour old; confirm the available price before using it. ' : ''}Estimates assume a two-outcome win market; draws, no contests and commissions follow the bookmaker’s rules.`;
        rememberJournalDraft();
        $('journal-selection').innerHTML = `<option value="">Choose a fighter</option><option value="fighter1">${escape(fighter1)}</option><option value="fighter2">${escape(fighter2)}</option>`;
        restoreJournalDraft(fighter1, fighter2);
        $('price-result').hidden = false;
    } catch (error) {
        message($('price-error'), error.message, true);
    }
});

function refreshJournal() {
    try {
        entries = readJournal();
        storageError = false;
        message($('journal-status'), '');
        renderJournal();
    } catch (error) {
        storageError = true;
        message($('journal-status'), `${error.message} Existing data has not been replaced.`, true);
    }
}
function writeJournal(transform) {
    const current = readJournal();
    entries = saveJournal(transform(current));
    storageError = false;
    message($('journal-status'), '');
    renderJournal();
}
$('save-research').addEventListener('submit', event => {
    event.preventDefault();
    try {
        if (!comparison) throw new Error('Compare the current prices before saving.');
        if (storageError) throw new Error('Resolve the journal storage error before saving another entry.');
        const selection = $('journal-selection').value;
        if (!['fighter1','fighter2'].includes(selection)) throw new Error('Choose which fighter you are recording.');
        const side = comparison.result.sides[selection === 'fighter1' ? 0 : 1];
        const createdAt = new Date().toISOString();
        const context = comparison.modelContext;
        const entry = {
            id:crypto.randomUUID(),createdAt,fighter1:comparison.fighter1,fighter2:comparison.fighter2,selection,
            decimalOdds:side.decimal,americanOdds:side.american,probability:side.modelProbability,
            probabilitySource:comparison.probabilitySource,stake:Number($('journal-stake').value),bookmaker:comparison.bookmaker,
            quotedAt:comparison.quotedAt,notes:$('journal-notes').value.trim(),outcome:'open',settledAt:null,profit:null,
            modelVersion:comparison.probabilitySource === 'model' ? context?.modelVersion ?? null : null,
            modelDataThrough:comparison.probabilitySource === 'model' ? context?.dataThrough ?? null : null,
            quoteSource:comparison.quoteSource,eventStartTime:comparison.eventStartTime,forecastContext:context?.forecastContext ?? null,
        };
        writeJournal(current => [entry,...current]);
        $('journal-selection').value = '';
        $('journal-notes').value = '';
        rememberJournalDraft();
        message($('save-status'),'Saved in this browser. The quoted price and probability are preserved.');
    } catch (error) {
        message($('save-status'),error.message,true);
    }
});

function renderJournal() {
    $('journal-count').textContent = entries.length ? `(${entries.length})` : '';
    $('export-journal').disabled = !entries.length;
    $('journal-summary').hidden = !entries.length;
    const summary = journalSummary(entries);
    const cards = [
        ['Open',String(summary.open)],
        ['Net result',summary.settled ? `${signed(summary.profit)} u` : '—'],
        ['Return on settled stake',summary.roi === null ? '—' : signed(summary.roi * 100,1) + '%'],
        ['Win rate',summary.winRate === null ? '—' : percent(summary.winRate)],
    ];
    $('journal-summary').innerHTML = cards.map(([label,value]) => `<div><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`).join('');
    $('journal-entries').innerHTML = entries.length ? entries.map(entry => {
        const selected = entry[entry.selection];
        return `<article class="journal-entry" data-entry="${escape(entry.id)}">
            <div class="journal-entry-top"><div><h3>${escape(selected)} <span>${american(entry.americanOdds)}</span></h3><p>${escape(entry.fighter1)} vs ${escape(entry.fighter2)} · ${escape(entry.bookmaker)}</p></div><strong class="entry-result">${entry.outcome === 'open' ? 'Open' : `${signed(entry.profit)} u`}</strong></div>
            <div class="entry-facts"><span>${entry.stake} u staked</span><span>${percent(entry.probability)} ${entry.probabilitySource === 'user' ? 'your estimate' : 'research model'}</span><span>Saved ${escape(dateTime(entry.createdAt))}</span></div>
            ${entry.notes ? `<p class="entry-notes">${escape(entry.notes)}</p>` : ''}
            <details class="entry-details"><summary>Snapshot details</summary><p>Quoted ${escape(dateTime(entry.quotedAt))} · ${entry.quoteSource === 'the_odds_api' ? 'The Odds API' : 'User-entered quote'}${entry.eventStartTime ? ` · Event ${escape(dateTime(entry.eventStartTime))}` : ''}</p>${entry.modelVersion ? `<p>Model ${escape(entry.modelVersion.slice(0,19))} · training data through ${escape(day(entry.modelDataThrough))}</p>` : ''}${entry.forecastContext ? `<p>Hypothetical fight: ${escape(day(entry.forecastContext.eventDate))} · ${escape(entry.forecastContext.weightClass)} · ${entry.forecastContext.rounds} rounds · ${entry.forecastContext.titleBout ? 'title bout' : 'non-title bout'}. Calibrated through ${escape(day(entry.forecastContext.calibrationThrough))}; research data through ${escape(day(entry.forecastContext.researchDataThrough))}.</p>` : ''}<p>Original odds, estimate and stake are preserved. Outcomes below are self-reported.</p></details>
            <div class="entry-actions"><label>Result<select data-outcome="${escape(entry.id)}" aria-label="Result for ${escape(selected)}"><option value="open" ${entry.outcome === 'open' ? 'selected' : ''}>Open</option><option value="won" ${entry.outcome === 'won' ? 'selected' : ''}>Won</option><option value="lost" ${entry.outcome === 'lost' ? 'selected' : ''}>Lost</option><option value="void" ${entry.outcome === 'void' ? 'selected' : ''}>Void / returned</option></select></label><button type="button" class="quiet-button remove-entry" data-remove="${escape(entry.id)}">Remove</button></div>
        </article>`;
    }).join('') : '<div class="journal-empty"><h3>No decisions saved yet</h3><p>Save a price comparison with your reasoning, then return here to record the outcome.</p><a href="#price-research" class="quiet-button">Compare a price</a></div>';
}
$('journal-entries').addEventListener('change', event => {
    const id = event.target.dataset.outcome;
    if (!id) return;
    try {
        const outcome = event.target.value;
        writeJournal(current => current.map(entry => entry.id !== id ? entry : outcome === 'open'
            ? {...entry,outcome:'open',settledAt:null,profit:null} : settleEntry(entry,outcome)));
    } catch (error) {
        message($('journal-status'),error.message,true);
        renderJournal();
    }
});
$('journal-entries').addEventListener('click', event => {
    const button = event.target.closest('[data-remove]');
    if (!button) return;
    try {
        let removed;
        writeJournal(current => {
            removed = current.find(entry => entry.id === button.dataset.remove);
            return current.filter(entry => entry.id !== button.dataset.remove);
        });
        lastRemoved = removed;
        $('journal-undo').hidden = !lastRemoved;
    } catch (error) { message($('journal-status'),error.message,true); }
});
$('undo-remove').addEventListener('click', () => {
    if (!lastRemoved) return;
    try {
        writeJournal(current => current.some(entry => entry.id === lastRemoved.id) ? current : [lastRemoved,...current]);
        lastRemoved = null;
        $('journal-undo').hidden = true;
    } catch (error) { message($('journal-status'),error.message,true); }
});
$('export-journal').addEventListener('click', () => {
    try {
        const data = readJournal();
        if (!data.length) return;
        const columns = Object.keys(data[0]);
        const csvValue = value => {
            let text = value && typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
            if (typeof value === 'string' && /^[=+@\-\t\r]/.test(text)) text = `'${text}`;
            return `"${text.replaceAll('"','""')}"`;
        };
        const csv = [columns.map(csvValue).join(','),...data.map(entry => columns.map(key => csvValue(entry[key])).join(','))].join('\r\n');
        const url = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = `versus-journal-${new Date().toISOString().slice(0,10)}.csv`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url),1000);
    } catch (error) { message($('journal-status'),error.message,true); }
});
window.addEventListener('storage',event => { if (event.key === KEY) refreshJournal(); });
refreshJournal();

$('load-prices').addEventListener('click',async () => {
    $('price-tools').open = true;
    $('load-prices').disabled = true;
    message($('feed-status'),'Loading available moneylines…');
    try {
        const data = await getJSON('/api/odds');
        feedEvents = data.events;
        if (!data.configured || !feedEvents.length) {
            $('feed-results').replaceChildren();
            message($('feed-status'),data.configured ? 'No upcoming two-outcome markets were returned. You can enter a quote manually.' : 'No price feed is connected. Enter a quote manually.');
            return;
        }
        message($('feed-status'),`The Odds API · retrieved ${dateTime(data.fetchedAt)}. Check the quote time; prices can change.`);
        $('feed-results').innerHTML = feedEvents.map((event,index) => `<details class="feed-event"><summary>${escape(event.homeTeam)} vs ${escape(event.awayTeam)}<span>${escape(dateTime(event.startTime))}</span></summary><div class="market-table-scroll"><table class="feed-table"><thead><tr><th>Bookmaker</th><th>${escape(event.homeTeam)}</th><th>${escape(event.awayTeam)}</th><th>Quoted</th><th></th></tr></thead><tbody>${event.bookmakers.map((book,bookIndex) => `<tr><th scope="row">${escape(book.title)}</th><td>${book.outcomes.find(outcome => outcome.name === event.homeTeam).price.toFixed(2)}</td><td>${book.outcomes.find(outcome => outcome.name === event.awayTeam).price.toFixed(2)}</td><td>${escape(dateTime(book.lastUpdate))}</td><td><button class="quiet-button" type="button" data-quote="${index}:${bookIndex}">Use quote</button></td></tr>`).join('')}</tbody></table></div></details>`).join('');
    } catch {
        message($('feed-status'),'Prices are unavailable. You can still enter a bookmaker quote manually.',true);
    } finally { $('load-prices').disabled = false; }
});
$('feed-results').addEventListener('click',event => {
    const button = event.target.closest('[data-quote]');
    if (!button) return;
    const [eventIndex,bookIndex] = button.dataset.quote.split(':').map(Number);
    const fight = feedEvents[eventIndex];
    const book = fight.bookmakers[bookIndex];
    $('fighter1').value = fight.homeTeam;
    $('fighter1').dispatchEvent(new Event('input',{bubbles:true}));
    $('fighter2').value = fight.awayTeam;
    $('fighter2').dispatchEvent(new Event('input',{bubbles:true}));
    $('odds-format').value = 'decimal';
    $('price-f1').inputMode = 'decimal';
    $('price-f2').inputMode = 'decimal';
    $('bookmaker').value = book.title;
    $('price-f1').value = book.outcomes.find(outcome => outcome.name === fight.homeTeam).price;
    $('price-f2').value = book.outcomes.find(outcome => outcome.name === fight.awayTeam).price;
    $('quoted-at').value = localTime(book.lastUpdate);
    providerQuotedAt = book.lastUpdate;
    quoteSource = 'the_odds_api';
    eventStartTime = fight.startTime;
    syncLabels();
    renderEstimateContext();
    message($('feed-status'),`Loaded ${book.title} prices. Enter your probability estimate, or analyze the matchup if both fighters are in the model.`);
    $('price-tools').open = true;
    $('price-research').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
});
