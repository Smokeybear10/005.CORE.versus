const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const date = value => value ? new Date(`${value.slice(0,10)}T12:00:00Z`).toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'}) : 'Date unavailable';
const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const percentage = value => Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : '—';
const resultLabel = {win:'Win',loss:'Loss',draw:'Draw',no_contest:'No contest',unknown:'Unrecorded'};
const sourceLink = fight => /^https?:\/\/ufcstats\.com\/fight-details\/[a-f0-9]{16}$/.test(fight.url || '')
    ? `<a href="${escape(fight.url)}" target="_blank" rel="noopener noreferrer" aria-label="Fight statistics: ${escape(fight.opponent)}, ${escape(date(fight.date))}">Stats ↗</a>` : '';
const record = value => `${value.wins}W · ${value.losses}L${value.draws ? ` · ${value.draws}D` : ''}${value.noContests ? ` · ${value.noContests}NC` : ''}${value.unknown ? ` · ${value.unknown} unknown` : ''}`;
let requestId = 0;
let selected = null;
let researchController = null;

function fightRow(fight) {
    const method = `${fight.method || 'Method unavailable'}${fight.resultUncertainty ? ' · Round disputed' : fight.round ? ` · R${fight.round}` : ''}`;
    return `<li class="fight-row"><span class="fight-result result-${escape(fight.result)}">${escape(resultLabel[fight.result] || 'Unrecorded')}</span><div class="fight-row-main"><strong>${escape(fight.opponent)}</strong><p class="fight-row-meta">${escape(date(fight.date))} · ${escape(method)}</p>${fight.promotion ? `<p class="fight-row-promotion">${escape(fight.promotion)}</p>` : ''}${fight.resultUncertainty ? `<p class="fight-uncertainty">${escape(fight.resultUncertainty)}</p>` : ''}</div>${fight.url ? `<span class="fight-row-source">${sourceLink(fight)}</span>` : ''}</li>`;
}

function renderForm(fighters) {
    $('fighter-form').innerHTML = fighters.map((fighter, index) => `
        <article class="form-column side-${index}">
            <header class="form-identity">
                <span class="eyebrow">${escape(fighter.latestBout?.weightClass || (index === 0 ? 'First fighter' : 'Second fighter'))}</span>
                <h3>${escape(fighter.name)}</h3>
            </header>
            <div class="form-recent">
                <div><span class="form-record-label">Last ${escape(fighter.recentFive.count)} UFC ${fighter.recentFive.count === 1 ? 'bout' : 'bouts'}</span><strong class="form-record">${escape(record(fighter.recentFive.record))}</strong></div>
                <div class="form-results" role="group" aria-label="Recent results, newest first">${fighter.recentFive.fights.map(fight => `<span class="form-result result-${escape(fight.result)}" title="${escape(resultLabel[fight.result])} vs ${escape(fight.opponent)}, ${escape(date(fight.date))}">${escape({win:'W',loss:'L',draw:'D',no_contest:'NC'}[fight.result] || '—')}</span>`).join('')}</div>
            </div>
            <dl class="form-facts">
                <div><dt>UFC record in dataset</dt><dd>${escape(record(fighter.record))}</dd></div>
                <div class="form-latest"><dt>Latest bout</dt><dd>${escape(date(fighter.latestBout?.date))}</dd></div>
            </dl>
            ${fighter.careerHistory?.available && fighter.careerHistory.totalRecordedBouts ? `<p class="research-muted form-history"><a href="#career-history-section">${escape(fighter.careerHistory.totalRecordedBouts)} earlier or non-UFC bouts</a></p>` : ''}
        </article>`).join('');
}

function renderFindings(data) {
    const [first, second] = data.fighters;
    const findings = [];
    const recentWins = data.fighters.map(fighter => {
        const recent = fighter.recentFive;
        const methods = recent.winMethods;
        const finishes = [[methods.koTko,'KO/TKO'],[methods.submission,'submission'],[methods.decision,'decision'],[methods.other,'other methods']].filter(([count]) => count).map(([count,method]) => `${count} by ${method}`).join(', ');
        return `${fighter.name}: ${finishes || (recent.count ? 'no wins' : 'no recorded bouts')} in the last ${recent.count} UFC bouts.`;
    });
    findings.push({label:'Recent wins',text:recentWins.join(' ')});
    const a = first.recentMetrics;
    const b = second.recentMetrics;
    if (a?.available && b?.available && [a.values.sigLandedPerMinute,a.values.sigAbsorbedPerMinute,b.values.sigLandedPerMinute,b.values.sigAbsorbedPerMinute].every(Number.isFinite)) {
        const aNet = a.values.sigLandedPerMinute - a.values.sigAbsorbedPerMinute;
        const bNet = b.values.sigLandedPerMinute - b.values.sigAbsorbedPerMinute;
        const signed = value => `${value > 0 ? '+' : ''}${number(value)}`;
        findings.push({label:'Striking balance',text:`Landed minus absorbed per minute: ${first.name} ${signed(aNet)}, ${second.name} ${signed(bNet)}. Based on ${a.coveredBouts} and ${b.coveredBouts} measured bouts against different opponents.`});
    }
    if (data.headToHead.total) findings.push({label:'Previous meetings',text:`${data.headToHead.total} recorded UFC ${data.headToHead.total === 1 ? 'meeting' : 'meetings'}. Those results are included in this report; this is not a forecast made before them.`});
    else findings.push({label:'Previous meetings',text:'No UFC meeting appears in this dataset. The comparison is based on their performances against other opponents.'});
    $('matchup-findings').innerHTML = findings.map(item => `<article><h4>${escape(item.label)}</h4><p>${escape(item.text)}</p></article>`).join('');
}

function renderCareerHistory(fighters) {
    const histories = fighters.map(fighter => fighter.careerHistory);
    $('career-history-section').hidden = !histories.some(history => history?.available);
    if ($('career-history-section').hidden) return;
    $('career-history').innerHTML = fighters.map(fighter => {
        const history = fighter.careerHistory;
        if (!history?.available) return `<article><h4>${escape(fighter.name)}</h4><p class="research-muted">Additional history is unavailable.</p></article>`;
        if (!history.totalRecordedBouts) return `<article><h4>${escape(fighter.name)}</h4><p class="research-muted">No additional results found in this source.</p></article>`;
        const methods = history.winMethods;
        const finishes = [[methods.koTko,'KO/TKO'],[methods.submission,'submission'],[methods.decision,'decision']]
            .filter(([count]) => count).map(([count,method]) => `${count} ${method}`).join(' · ');
        const promotions = history.promotions.slice(0, 5).map(item => `${item.name || 'Unrecorded promotion'} (${item.bouts})`).join(' · ');
        return `<article><h4>${escape(fighter.name)}</h4><p class="career-record">${escape(record(history.record))}</p><p class="research-muted">${history.totalRecordedBouts} additional bouts${finishes ? ` · Wins: ${escape(finishes)}` : ''}</p>${promotions ? `<p class="research-muted">${escape(promotions)}</p>` : ''}<details class="method-note"><summary>View ${history.recentBouts.length === history.totalRecordedBouts ? 'recorded' : `latest ${history.recentBouts.length}`} results</summary><ol class="recent-fight-list">${history.recentBouts.map(fightRow).join('')}</ol></details></article>`;
    }).join('');
    const source = histories.find(history => history?.available)?.source;
    $('career-source').textContent = `Source: ${source?.name || 'Fight Canon'}. Results recorded through ${date(histories.find(history => history?.available)?.asOf)}. Coverage varies by fighter; this is not a complete career record.`;
    if (source?.url && /^https:\/\/github\.com\/Jadevit\/fight-canon-db(?:\/|$)/.test(source.url)) {
        const link = document.createElement('a');
        link.href = source.url;
        link.textContent = ' Source snapshot ↗';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        $('career-source').append(link);
    }
}

function metricSample(metric, key) {
    if (!metric?.available) return '';
    const totals = metric.totals;
    const samples = {
        sigAccuracy: [totals.sigLanded, totals.sigAttempted, 'landed'],
        sigDefense: [totals.sigAbsorbed, totals.opponentSigAttempted, 'landed against'],
        tdAccuracy: [totals.tdLanded, totals.tdAttempted, 'landed'],
        tdDefense: [totals.opponentTdLanded, totals.opponentTdAttempted, 'landed against'],
    };
    const sample = samples[key];
    return sample ? `<small class="metric-sample">${sample[0]} of ${sample[1]} ${sample[2]}</small>` : '';
}

function metricCell(metrics, index, key, format) {
    const value = metrics[index]?.available ? metrics[index].values?.[key] : null;
    const values = metrics.map(metric => metric?.available ? metric.values?.[key] : null);
    const maximum = Math.max(0, ...values.filter(Number.isFinite));
    const width = Number.isFinite(value) && maximum > 0 ? Math.max(0, Math.min(100, value / maximum * 100)) : 0;
    return `<td class="metric-cell side-${index}"><span class="metric-value">${format(value)}</span>${values.every(Number.isFinite) ? `<span class="metric-bar" style="--metric-width:${width.toFixed(2)}%" aria-hidden="true"><span></span></span>` : ''}${metricSample(metrics[index],key)}</td>`;
}

function renderMetrics(fighters) {
    const metrics = fighters.map(fighter => fighter.recentMetrics);
    $('fight-metrics-section').hidden = !metrics.some(metric => metric?.available);
    if ($('fight-metrics-section').hidden) return;
    const rows = [
        ['Significant strikes landed / min','sigLandedPerMinute','Landed strikes, divided by measured fight time.',number],
        ['Significant strikes absorbed / min','sigAbsorbedPerMinute','Opponent’s landed strikes, divided by measured fight time.',number],
        ['Striking accuracy','sigAccuracy','Landed / attempted significant strikes.',percentage],
        ['Striking defense','sigDefense','Opponent’s missed / attempted significant strikes.',percentage],
        ['Takedowns landed / 15 min','tdLandedPer15','Successful takedowns, normalized to 15 minutes.',number],
        ['Takedown accuracy','tdAccuracy','Successful / attempted takedowns.',percentage],
        ['Takedown defense','tdDefense','Opponent’s failed / attempted takedowns.',percentage],
    ];
    $('fight-metrics-note').textContent = fighters.map((fighter,index) => `${fighter.name}: ${metrics[index]?.coveredBouts || 0}/${metrics[index]?.windowBouts || fighter.recentFive.count} recent bouts measured`).join(' · ');
    $('fight-metrics').innerHTML = `<table class="evidence-table metrics-table"><thead><tr><th scope="col">Measure</th>${fighters.map(fighter => `<th scope="col">${escape(fighter.name)}</th>`).join('')}</tr></thead><tbody>${rows.map(([label,key,definition,format]) => `<tr><th scope="row">${escape(label)}<small>${escape(definition)}</small></th>${metrics.map((_,index) => metricCell(metrics,index,key,format)).join('')}</tr>`).join('')}</tbody></table><p class="metric-key">Bars compare the values in each row. Longer is not always better; these are recorded statistics, not win probabilities.</p><details class="method-note"><summary>Measurement sample and definitions</summary>${fighters.map((fighter,index) => { const metric = metrics[index]; return `<p><strong>${escape(fighter.name)}</strong>: ${metric?.available ? `${number(metric.totalSeconds / 60,1)} minutes across ${metric.coveredBouts} bouts, ${escape(date(metric.from))}–${escape(date(metric.through))}. Significant strikes: ${metric.totals.sigLanded}/${metric.totals.sigAttempted}; opponents: ${metric.totals.sigAbsorbed}/${metric.totals.opponentSigAttempted}. Takedowns: ${metric.totals.tdLanded}/${metric.totals.tdAttempted}; opponents: ${metric.totals.opponentTdLanded}/${metric.totals.opponentTdAttempted}.` : 'Round statistics unavailable.'}</p>`; }).join('')}<p>A dash means unavailable or no attempts. Rates use summed counts and measured time, not averages of percentages. Different opponents and sample sizes limit direct comparisons.</p></details>`;
}

function renderOpponents(data) {
    $('recent-fights').innerHTML = data.fighters.map(fighter => `<div><h4>${escape(fighter.name)}</h4><ol class="recent-fight-list">${fighter.recentFive.fights.map(fightRow).join('')}</ol></div>`).join('');
    $('evidence-h2h').innerHTML = data.headToHead.total
        ? `<p class="research-muted">Results from ${escape(data.fighters[0].name)}’s side.</p><ol class="recent-fight-list">${data.headToHead.fights.map(fightRow).join('')}</ol>`
        : '<p class="research-muted">No previous UFC meeting is recorded.</p>';
    $('common-opponents').innerHTML = data.commonOpponents.total
        ? `<div class="market-table-scroll"><table class="evidence-table common-table"><thead><tr><th scope="col">Opponent</th>${data.fighters.map(fighter => `<th scope="col">${escape(fighter.name)}</th>`).join('')}</tr></thead><tbody>${data.commonOpponents.opponents.map(opponent => `<tr><th scope="row">${escape(opponent.name)}</th>${[opponent.firstFighter,opponent.secondFighter].map(side => `<td>${side.fights.map(fight => `<div><strong>${escape(resultLabel[fight.result])}</strong> · ${escape(fight.method)}<small>${escape(date(fight.date))} · ${sourceLink(fight)}</small></div>`).join('')}${side.total > side.fights.length ? `<small>Latest ${side.fights.length} of ${side.total} meetings shown</small>` : ''}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`
        : '<p class="research-muted">No shared opponents appear in the recorded UFC history.</p>';
}

async function researchMatchup(fighters) {
    researchController?.abort();
    researchController = new AbortController();
    selected = fighters;
    const currentRequest = ++requestId;
    $('matchup-research').hidden = false;
    $('matchup-research').setAttribute('aria-busy','true');
    $('matchup-heading').textContent = 'Matchup report';
    $('evidence-coverage').textContent = '';
    $('evidence-content').hidden = true;
    $('retry-evidence').hidden = true;
    $('evidence-status').textContent = 'Loading recorded fights and matchup evidence…';
    $('evidence-status').classList.remove('research-error');
    window.dispatchEvent(new CustomEvent('versus:research-loading', {detail:fighters}));
    try {
        const query = new URLSearchParams({fighter1:fighters.fighter1,fighter2:fighters.fighter2});
        const response = await fetch(`/api/matchup-research?${query}`, {signal:AbortSignal.any([researchController.signal, AbortSignal.timeout(55000)])});
        const data = await response.json();
        if (currentRequest !== requestId) return;
        if (!response.ok) throw new Error(data.error || 'Matchup research could not be loaded.');
        renderForm(data.fighters);
        renderFindings(data);
        renderMetrics(data.fighters);
        renderOpponents(data);
        renderCareerHistory(data.fighters);
        $('evidence-coverage').textContent = `Recorded UFC results through ${date(data.dataThrough)}.`;
        if (data.source?.sourceUrl && /^https:\/\/github\.com\/Greco1899\/scrape_ufc_stats(?:\/|$)/.test(data.source.sourceUrl)) {
            const link = document.createElement('a');
            link.href = data.source.sourceUrl;
            link.textContent = ' Data source ↗';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            $('evidence-coverage').append(link);
        }
        $('evidence-gaps').innerHTML = data.gaps.map(gap => `<li>${escape(gap.text)}</li>`).join('');
        $('evidence-content').hidden = false;
        $('evidence-status').textContent = '';
        window.dispatchEvent(new CustomEvent('versus:evidence', {detail:data}));
    } catch (error) {
        if (currentRequest !== requestId) return;
        $('evidence-status').textContent = error.name === 'TimeoutError' ? 'Research took too long to load. Try again.' : error.message;
        $('evidence-status').classList.add('research-error');
        $('retry-evidence').hidden = false;
        window.dispatchEvent(new CustomEvent('versus:research-error', {detail:fighters}));
    } finally {
        if (currentRequest === requestId) $('matchup-research').setAttribute('aria-busy','false');
    }
}

window.addEventListener('versus:research-start', event => {
    $('model-panel').hidden = true;
    $('model-details').open = false;
    researchMatchup(event.detail);
});
window.addEventListener('versus:matchup-change', () => {
    requestId += 1;
    researchController?.abort();
    selected = null;
    $('matchup-research').hidden = true;
    $('matchup-research').setAttribute('aria-busy','false');
    $('model-panel').hidden = true;
});
$('retry-evidence').addEventListener('click', () => { if (selected) researchMatchup(selected); });
