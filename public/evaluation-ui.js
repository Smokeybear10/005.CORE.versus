const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const number = value => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
const date = value => {
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}/.test(value)) return 'Not recorded';
    const parsed = new Date(`${value.slice(0, 10)}T12:00:00Z`);
    return Number.isNaN(parsed.getTime()) ? 'Not recorded' : parsed.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
};
const element = (tag, text) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
};
const readJson = async url => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 55000);
    try {
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) throw new Error('Data unavailable');
        return await response.json();
    } finally {
        clearTimeout(timer);
    }
};
const setFields = (attribute, fields) => {
    document.querySelectorAll(`[${attribute}]`).forEach(node => {
        node.textContent = fields[node.getAttribute(attribute)] ?? '—';
    });
};

function renderPeriod(id, period) {
    const target = document.getElementById(id);
    if (!period?.from || !period?.through) {
        target.textContent = 'Not recorded';
        return;
    }
    target.textContent = `${date(period.from)} – ${date(period.through)}`;
    const count = period.decisiveBouts ?? period.bouts ?? period.count;
    if (Number.isFinite(count)) target.append(element('span', `${number(count)} fights`));
}

function renderCalibration(audit) {
    const bins = audit.calibration_bins ?? audit.calibrationBins;
    if (!Array.isArray(bins)) return;
    const rows = bins.filter(bin => Number.isFinite(bin.lower) && Number.isFinite(bin.upper) && Number.isInteger(bin.count) && bin.count >= 0).map(bin => {
        const row = element('tr');
        const range = element('th', `${Math.round(bin.lower * 100)}–${Math.round(bin.upper * 100)}%`);
        range.scope = 'row';
        row.append(range, element('td', number(bin.count)), element('td', percent(bin.mean_confidence ?? bin.predicted)), element('td', percent(bin.hit_rate ?? bin.observed)));
        if (!bin.count) row.className = 'results-no-sample';
        return row;
    });
    if (!rows.length) return;
    document.getElementById('calibration-rows').replaceChildren(...rows);
    const highConfidence = audit.threshold_subsets?.['0.8'] ?? audit.confidenceBands?.['0.8'];
    document.getElementById('calibration-note').textContent = Number.isInteger(highConfidence?.count)
        ? `${number(highConfidence.count)} replay fights reached at least 80% modeled confidence. This small sample does not establish reliable high-confidence performance. A dash means there were no predictions in that group.`
        : 'Small samples have wide uncertainty. A dash means there were no predictions in that group.';
    document.getElementById('calibration-section').hidden = false;
}

async function loadEvaluation() {
    const status = document.getElementById('confidence-audit');
    const retry = document.getElementById('evaluation-retry');
    if (!status) return;
    status.textContent = 'Loading the evaluation for the research model…';
    retry.hidden = true;
    document.getElementById('evaluation-generated').textContent = '';
    document.getElementById('evaluation-fingerprint').textContent = '';
    try {
        const data = await readJson('/api/evaluation');
        const candidate = data.candidate;
        const audit = candidate?.audit;
        if (!audit || !Number.isInteger(audit.count) || audit.count <= 0 || !Number.isFinite(audit.accuracy) || audit.accuracy < 0 || audit.accuracy > 1 || !Number.isFinite(audit.brier) || audit.brier < 0 || audit.brier > 1) throw new Error('Incomplete evaluation');
        const fields = { accuracy: percent(audit.accuracy), brier: audit.brier.toFixed(4), fightCount: number(audit.count), eventCount: number(audit.eventCount), featureCount: number(candidate.featureCount) };
        setFields('data-evaluation-field', fields);
        status.textContent = '';
        document.getElementById('evaluation-generated').textContent = data.generatedAt ? `Report generated ${date(data.generatedAt)}` : '';
        document.getElementById('evaluation-summary').hidden = false;
        const interval = audit.accuracyCI ?? audit.accuracy_wilson_95;
        const uncertainty = Array.isArray(interval) && interval.length === 2 && interval.every(Number.isFinite)
            ? `Accuracy’s 95% Wilson interval is ${percent(interval[0])}–${percent(interval[1])}. ` : '';
        document.getElementById('evaluation-uncertainty').textContent = `${uncertainty}These scores cover all eligible replay bouts before the app’s matchup restrictions. Brier score measures squared probability error; a constant 50% estimate scores 0.2500.`;
        renderCalibration(audit);
        renderPeriod('evaluation-training', candidate.trainedOn ?? candidate.training);
        renderPeriod('evaluation-calibration', candidate.calibration);
        const limitations = Array.isArray(data.limitations) ? data.limitations.filter(value => typeof value === 'string' && value.trim()) : [];
        if (limitations.length) document.getElementById('evaluation-limitations').replaceChildren(...limitations.map(value => element('li', value)));
        document.getElementById('evaluation-fingerprint').textContent = candidate.modelVersion ? `Model: ${candidate.modelVersion}` : '';
    } catch {
        status.textContent = 'The model evaluation is unavailable. Scores are withheld until a matching evaluation can be loaded.';
        retry.hidden = false;
        document.getElementById('evaluation-summary').hidden = true;
        document.getElementById('calibration-section').hidden = true;
        document.getElementById('evaluation-training').textContent = 'Unavailable';
        document.getElementById('evaluation-calibration').textContent = 'Unavailable';
        setFields('data-evaluation-field', {});
    }
}

async function loadCoverage() {
    const status = document.getElementById('research-info-status');
    if (!status) return;
    try {
        const { research } = await readJson('/api/research-status');
        if (!research?.available) throw new Error('Coverage unavailable');
        setFields('data-research-field', { dataThrough: date(research.dataThrough), fighterCount: number(research.fighterCount) });
        status.textContent = 'Loaded from the research dataset. Each matchup has its own usable sample.';
    } catch {
        setFields('data-research-field', { dataThrough: 'Unavailable', fighterCount: '—' });
        status.textContent = 'Current research coverage could not be loaded. The saved evaluation is shown separately.';
    }
}

let legacyLoaded = false;
async function loadLegacy() {
    const details = document.getElementById('legacy-evaluation');
    if (!details.open || legacyLoaded) return;
    legacyLoaded = true;
    const table = document.getElementById('model-evaluation');
    const status = document.getElementById('model-info-status');
    const labels = { logistic_regression: 'Logistic regression', random_forest: 'Random forest', gradient_boosting: 'Gradient boosting', ensemble: 'Ensemble mean' };
    try {
        const info = await readJson('/api/model-info');
        const rows = Object.entries(info.models ?? {}).map(([key, metrics]) => {
            const row = element('tr');
            const label = element('th', labels[key] ?? key);
            label.scope = 'row';
            row.append(label, element('td', percent(metrics.accuracy)), element('td', percent(metrics.ts_accuracy)));
            return row;
        });
        if (!rows.length) throw new Error('No archived scores');
        table.replaceChildren(...rows);
        status.textContent = 'Archived classifier scores. A matchup probability is different from classification accuracy.';
    } catch {
        const row = element('tr');
        const cell = element('td', 'Archived scores are unavailable.');
        cell.colSpan = 3;
        row.append(cell);
        table.replaceChildren(row);
        status.textContent = 'Close and reopen this section to try again.';
        legacyLoaded = false;
    }
}

document.getElementById('evaluation-retry')?.addEventListener('click', loadEvaluation);
document.getElementById('legacy-evaluation')?.addEventListener('toggle', loadLegacy);
loadEvaluation();
loadCoverage();

const revealArchive = () => {
    if (["#evaluation", "#legacy-evaluation"].includes(window.location.hash)) {
        document.getElementById("legacy-evaluation").open = true;
    }
};
window.addEventListener("hashchange", revealArchive);
revealArchive();
