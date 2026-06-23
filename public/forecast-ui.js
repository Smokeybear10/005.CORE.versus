const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const forecastPercent = value => value >= 0.8995 && value < 0.9 ? '&lt;90%' : percent(value);
const localDate = () => { const now = new Date(); return `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`; };
let matchup = null;
let status = null;
let report = null;
let revision = 0;
let forecastController = null;
let modelLoading = false;

async function getJSON(url, options = {}) {
    const timeout = AbortSignal.timeout(55000);
    const response = await fetch(url, {...options,signal:options.signal ? AbortSignal.any([options.signal,timeout]) : timeout});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The estimate could not be loaded.');
    return data;
}

function invalidate() {
    revision += 1;
    forecastController?.abort();
    forecastController = null;
    $('forecast-result').hidden = true;
    $('forecast-error').hidden = true;
    $('forecast-submit').disabled = false;
    $('forecast-submit').textContent = 'Estimate this matchup';
    $('forecast-form').setAttribute('aria-busy','false');
    window.dispatchEvent(new CustomEvent('versus:forecast-invalidated'));
}

function renderAudit() {
    if (!report?.candidate?.audit || report.candidate.modelVersion !== status?.modelVersion) return;
    const audit = report.candidate.audit;
    $('forecast-audit').hidden = false;
    $('forecast-audit-content').innerHTML = `<p>After correcting fighter identities, the model selected the winner in <strong>${percent(audit.accuracy)} of ${audit.count} historical fights</strong>. Training and calibration used earlier years. This overall accuracy is not the probability for a particular fight.</p><p>These outcomes were already examined during development. This replay is not independent validation, and high-confidence accuracy is not established.</p><div class="market-table-scroll"><table class="evidence-table"><thead><tr><th>Confidence band</th><th>Fights</th><th>Observed win rate</th></tr></thead><tbody>${(audit.calibration_bins ?? audit.calibrationBins ?? []).map(bin => `<tr><th>${Math.round(bin.lower * 100)}–${Math.round(bin.upper * 100)}%</th><td>${bin.count}</td><td>${percent(bin.hit_rate ?? bin.observedWinRate)}${(bin.wilson_95 || bin.observedCI) ? `<small>95% interval: ${(bin.wilson_95 || bin.observedCI).map(percent).join('–')}</small>` : ''}</td></tr>`).join('')}</tbody></table></div><p>Small groups have wide uncertainty. These results cover all eligible decisive bouts; performance after the app’s matchup restrictions has not been separately tested. This is not a verified betting record.</p>`;
}

function setup() {
    if (!matchup) return;
    $('forecast-panel').hidden = false;
    $('forecast-form').hidden = !status?.available;
    if (!status?.available) {
        $('forecast-availability').textContent = modelLoading
            ? 'Loading the model…'
            : 'The model is unavailable. You can still use the matchup report.';
        return;
    }
    const divisions = status.supportedWeightClasses || [];
    const recentDivisions = matchup.fighters.map(fighter => {
        const value = fighter.latestBout?.weightClass || '';
        return [...divisions].sort((first,second) => second.length - first.length).find(division => (value === division || value.endsWith(division) || value.endsWith(`${division} Tournament`)) && division.includes("Women's") === value.includes("Women's"));
    });
    const sharedDivision = recentDivisions[0] === recentDivisions[1] ? recentDivisions[0] : '';
    $('forecast-division').innerHTML = '<option value="">Choose a division</option>' + divisions.map(division => `<option value="${escape(division)}">${escape(division)}</option>`).join('');
    if (divisions.includes(sharedDivision)) $('forecast-division').value = sharedDivision;
    $('forecast-date').value = localDate();
    $('forecast-date').min = localDate();
    if (status.forecastDateRange?.max) $('forecast-date').max = status.forecastDateRange.max;
    $('forecast-availability').textContent = sharedDivision
        ? `Last recorded division: ${sharedDivision}. Confirm the fight setup below.`
        : 'No shared recent division. Select the intended division; unsupported matchups receive no estimate.';
    renderAudit();
}

async function loadModel() {
    if (modelLoading) return;
    modelLoading = true;
    setup();
    const results = await Promise.allSettled([getJSON('/api/forecast-status'), getJSON('/api/evaluation')]);
    status = results[0].status === 'fulfilled' ? results[0].value : null;
    report = results[1].status === 'fulfilled' ? results[1].value : null;
    modelLoading = false;
    setup();
}

loadModel();
window.addEventListener('versus:evidence', event => {
    invalidate();
    matchup = event.detail;
    if (!status?.available && !modelLoading) loadModel();
    setup();
});
window.addEventListener('versus:matchup-change', () => { invalidate(); matchup = null; $('forecast-panel').hidden = true; });
window.addEventListener('versus:research-start', () => { invalidate(); matchup = null; $('forecast-panel').hidden = true; });
$('forecast-title').addEventListener('change', () => { if ($('forecast-title').value === 'yes') $('forecast-rounds').value = '5'; });
$('forecast-rounds').addEventListener('change', () => { if ($('forecast-rounds').value === '3') $('forecast-title').value = 'no'; });
$('forecast-form').addEventListener('input', invalidate);
$('forecast-form').addEventListener('change', invalidate);
$('forecast-form').addEventListener('submit', async event => {
    event.preventDefault();
    invalidate();
    const currentRevision = revision;
    if (!matchup || !status?.available) return;
    const eventDate = $('forecast-date').value;
    const weightClass = $('forecast-division').value;
    if (!eventDate || eventDate < localDate() || (status.forecastDateRange?.max && eventDate > status.forecastDateRange.max) || !weightClass) {
        $('forecast-error').textContent = 'Choose a division and a date within the model’s supported date range.';
        $('forecast-error').hidden = false;
        return;
    }
    $('forecast-submit').disabled = true;
    $('forecast-submit').textContent = 'Estimating…';
    $('forecast-form').setAttribute('aria-busy','true');
    forecastController = new AbortController();
    try {
        const data = await getJSON('/api/forecast', {method:'POST',signal:forecastController.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({
            firstID:matchup.fighters[0].fighterId,secondID:matchup.fighters[1].fighterId,eventDate,weightClass,
            rounds:Number($('forecast-rounds').value),titleBout:$('forecast-title').value === 'yes',
        })});
        if (currentRevision !== revision) return;
        if (data.status !== 'forecast') {
            $('forecast-result').innerHTML = `<h3>No estimate for this matchup</h3><p>${escape((data.reasons || []).join(' ') || 'There is not enough supported fight history for this estimate.')}</p>`;
        } else {
            const estimateDate = new Date(`${data.context.eventDate}T12:00:00Z`).toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'});
            $('forecast-result').innerHTML = `<p class="forecast-context">${escape(data.context.weightClass)} · ${data.context.scheduledRounds} rounds${data.context.isTitleBout ? ' · Title bout' : ''} · ${escape(estimateDate)}</p><div class="forecast-probabilities">${[data.fighter1,data.fighter2].map((fighter,index) => `<div class="side-${index}"><h3>${escape(fighter.name)}</h3><strong>${forecastPercent(fighter.probability)}</strong><span>Estimated win probability</span><small>${fighter.validStatBouts} prior bouts with usable statistics</small></div>`).join('')}</div><div class="forecast-probability-track" aria-hidden="true"><span class="side-0" style="width:${Math.max(0,Math.min(100,data.fighter1.probability * 100))}%"></span><span class="side-1"></span></div><p class="forecast-limit">Conditional on a decisive result and this fight setup. The model has not demonstrated an advantage over market prices; its historical results are below.</p>${data.missingInputs?.length ? `<p class="research-muted">${data.missingInputs.length} missing measurements use training-period replacement values.</p>` : ''}<div class="research-next"><a href="#price-research" class="research-button" data-use-forecast>Compare this estimate with a price <span aria-hidden="true">→</span></a><p class="research-muted">Uses this probability and fight setup.</p></div>`;
            window.dispatchEvent(new CustomEvent('versus:forecast', {detail:{fighter1:data.fighter1.name,fighter2:data.fighter2.name,f1Prob:data.fighter1.probability,f2Prob:data.fighter2.probability,research:{modelVersion:data.modelVersion,dataThrough:data.provenance.trainingThrough,forecastContext:{eventDate:data.context.eventDate,weightClass:data.context.weightClass,rounds:data.context.scheduledRounds,titleBout:data.context.isTitleBout,trainingThrough:data.provenance.trainingThrough,calibrationThrough:data.provenance.calibrationThrough,researchDataThrough:data.provenance.researchDataThrough}}}}));
        }
        $('forecast-result').hidden = false;
    } catch (error) {
        if (currentRevision !== revision) return;
        $('forecast-error').textContent = error.name === 'TimeoutError' ? 'The estimate took too long. Try again.' : error.message;
        $('forecast-error').hidden = false;
    } finally {
        if (currentRevision === revision) {
            forecastController = null;
            $('forecast-submit').disabled = false;
            $('forecast-submit').textContent = 'Estimate this matchup';
            $('forecast-form').setAttribute('aria-busy','false');
        }
    }
});
