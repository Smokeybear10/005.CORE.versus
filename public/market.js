export const KEY = "versus.research.v1";

const MAX_ENTRIES = 500;
const ENTRY_FIELDS = [
    "id", "createdAt", "fighter1", "fighter2", "selection", "decimalOdds", "americanOdds",
    "probability", "probabilitySource", "stake", "bookmaker", "quotedAt", "notes", "outcome",
    "settledAt", "profit", "modelVersion", "modelDataThrough",
];
const OPTIONAL_FIELDS = ["quoteSource", "eventStartTime", "forecastContext"];

function numberInput(raw) {
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
    if (typeof raw !== "string" || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(raw.trim())) {
        throw new Error("Enter complete numeric odds, such as -110, +150, or 2.50.");
    }
    const value = Number(raw.trim());
    if (!Number.isFinite(value)) throw new Error("Odds must be a finite number.");
    return value;
}

function validateProbability(value) {
    if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error("Probability must be a number between 0 and 1.");
    }
}

export function parseOdds(raw, format = "american") {
    if (format !== "american" && format !== "decimal") {
        throw new Error("Choose American or decimal odds.");
    }
    const value = numberInput(raw);
    if (format === "american" && Math.abs(value) < 100) {
        throw new Error("American odds must be at least +100 or at most -100.");
    }
    const decimal = format === "decimal" ? value : value > 0 ? 1 + value / 100 : 1 + 100 / -value;
    if (!Number.isFinite(decimal) || decimal <= 1 || decimal > 10000) {
        throw new Error("Decimal odds must be greater than 1 and no higher than 10,000.");
    }
    const american = format === "american" ? value : decimal >= 2 ? (decimal - 1) * 100 : -100 / (decimal - 1);
    return { decimal, american, impliedProbability: 1 / decimal };
}

export function compareMarket({ odds1, odds2, format = "american", probability1 }) {
    validateProbability(probability1);
    const odds = [parseOdds(odds1, format), parseOdds(odds2, format)];
    const impliedTotal = odds[0].impliedProbability + odds[1].impliedProbability;
    return {
        sides: odds.map(({ decimal, american, impliedProbability }, index) => {
            const fairProbability = impliedProbability / impliedTotal;
            const modelProbability = index === 0 ? probability1 : 1 - probability1;
            return {
                decimal,
                american,
                breakEven: impliedProbability,
                fairProbability,
                modelProbability,
                gap: modelProbability - fairProbability,
                expectedReturn: modelProbability * decimal - 1,
            };
        }),
        overround: impliedTotal - 1,
    };
}

function validDate(value) {
    return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value)
        && Number.isFinite(Date.parse(`${value}T00:00:00Z`))
        && new Date(`${value}T00:00:00Z`).toISOString().slice(0, 10) === value;
}

function validTimestamp(value) {
    return typeof value === "string"
        && /^\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,3})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/.test(value)
        && validDate(value.slice(0, 10)) && Number.isFinite(Date.parse(value));
}

function validateText(value, label, maxLength) {
    if (typeof value !== "string" || !value.trim() || value.length > maxLength) {
        throw new Error(`${label} must contain between 1 and ${maxLength} characters.`);
    }
}

function expectedProfit(entry, outcome) {
    if (outcome === "won") return entry.stake * (entry.decimalOdds - 1);
    if (outcome === "lost") return -entry.stake;
    return 0;
}

function approximatelyEqual(first, second) {
    return Math.abs(first - second) <= 1e-10 * Math.max(1, Math.abs(first), Math.abs(second));
}

function validateEntry(entry) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)
        || Object.keys(entry).some(field => !ENTRY_FIELDS.includes(field) && !OPTIONAL_FIELDS.includes(field))
        || !ENTRY_FIELDS.every(field => Object.hasOwn(entry, field))) {
        throw new Error("A journal entry is missing required fields or contains unrecognized fields.");
    }
    const quoteSource = Object.hasOwn(entry, "quoteSource") ? entry.quoteSource : "manual";
    const eventStartTime = Object.hasOwn(entry, "eventStartTime") ? entry.eventStartTime : null;
    if (!["manual", "the_odds_api"].includes(quoteSource)) throw new Error("Quote source must be manual or the_odds_api.");
    if (eventStartTime !== null && !validTimestamp(eventStartTime)) {
        throw new Error("The event start time must be an ISO timestamp or null.");
    }
    const forecastContext = entry.forecastContext ?? null;
    if (forecastContext !== null) {
        const fields = ['eventDate','weightClass','rounds','titleBout','trainingThrough','calibrationThrough','researchDataThrough'];
        if (typeof forecastContext !== 'object' || Array.isArray(forecastContext)
            || Object.keys(forecastContext).length !== fields.length || !fields.every(field => Object.hasOwn(forecastContext, field))) {
            throw new Error('The saved forecast context is incomplete.');
        }
        for (const field of ['eventDate','trainingThrough','calibrationThrough','researchDataThrough']) {
            if (!validDate(forecastContext[field])) throw new Error('Forecast dates must be valid ISO dates.');
        }
        validateText(forecastContext.weightClass, 'Forecast division', 100);
        if (![3,5].includes(forecastContext.rounds) || typeof forecastContext.titleBout !== 'boolean') throw new Error('The saved forecast fight setup is invalid.');
        if (forecastContext.trainingThrough > forecastContext.calibrationThrough || forecastContext.calibrationThrough > forecastContext.researchDataThrough
            || forecastContext.researchDataThrough > forecastContext.eventDate) throw new Error('The saved forecast dates are out of order.');
        if (entry.probabilitySource !== 'model') throw new Error('Only a model estimate can have forecast context.');
    }
    const validated = { ...entry, quoteSource, eventStartTime, forecastContext };
    validateText(entry.id, "Entry ID", 128);
    validateText(entry.fighter1, "First fighter", 160);
    validateText(entry.fighter2, "Second fighter", 160);
    validateText(entry.bookmaker, "Bookmaker", 160);
    if (entry.fighter1.trim().toLowerCase() === entry.fighter2.trim().toLowerCase()) {
        throw new Error("A journal entry must contain two different fighters.");
    }
    if (!["fighter1", "fighter2"].includes(entry.selection)) throw new Error("Choose a fighter for the journal entry.");
    if (!["model", "user"].includes(entry.probabilitySource)) throw new Error("Probability source must be model or user.");
    validateProbability(entry.probability);
    if (!Number.isFinite(entry.decimalOdds) || !Number.isFinite(entry.americanOdds)) {
        throw new Error("Saved odds must be numeric.");
    }
    const decimalOdds = parseOdds(entry.decimalOdds, "decimal");
    const americanOdds = parseOdds(entry.americanOdds);
    if (!approximatelyEqual(decimalOdds.decimal, americanOdds.decimal)) {
        throw new Error("Saved American and decimal odds do not match.");
    }
    if (!Number.isFinite(entry.stake) || entry.stake <= 0 || !Number.isFinite(entry.stake * entry.decimalOdds)) {
        throw new Error("Stake must be a positive finite number with a finite potential return.");
    }
    if (!validTimestamp(entry.createdAt) || !validTimestamp(entry.quotedAt)) {
        throw new Error("Entry and quote times must be valid ISO timestamps with a time zone.");
    }
    if (Date.parse(entry.quotedAt) > Date.parse(entry.createdAt)) {
        throw new Error("The quote time cannot be later than the saved entry time.");
    }
    if (typeof entry.notes !== "string" || entry.notes.length > 2000) {
        throw new Error("Notes must be text no longer than 2,000 characters.");
    }
    if (entry.modelVersion !== null) validateText(entry.modelVersion, "Model version", 160);
    if (entry.modelDataThrough !== null && !validDate(entry.modelDataThrough)) {
        throw new Error("The model data date must be an ISO date or null.");
    }
    if (!["open", "won", "lost", "void"].includes(entry.outcome)) throw new Error("The saved outcome is invalid.");
    if (entry.outcome === "open") {
        if (entry.settledAt !== null || entry.profit !== null) {
            throw new Error("An open entry cannot have a settlement time or profit.");
        }
        return validated;
    }
    if (!validTimestamp(entry.settledAt) || Date.parse(entry.settledAt) < Date.parse(entry.createdAt)) {
        throw new Error("The settlement time must be a valid ISO timestamp at or after the entry time.");
    }
    if (!Number.isFinite(entry.profit) || !approximatelyEqual(entry.profit, expectedProfit(entry, entry.outcome))) {
        throw new Error("The saved profit does not match its outcome, odds, and stake.");
    }
    return validated;
}

function validateJournal(entries) {
    if (!Array.isArray(entries)) throw new Error("The journal must contain an array of entries.");
    if (entries.length > MAX_ENTRIES) throw new Error("The journal can hold at most 500 entries. Export or remove entries before saving more.");
    const ids = new Set();
    return Array.from(entries, entry => {
        const validated = validateEntry(entry);
        if (ids.has(entry.id)) throw new Error("The journal contains duplicate entry IDs.");
        ids.add(entry.id);
        return validated;
    });
}

export function settleEntry(entry, outcome) {
    const validated = validateEntry(entry);
    if (!["won", "lost", "void"].includes(outcome)) throw new Error("Choose won, lost, or void to record a result.");
    const settled = { ...validated, outcome, settledAt: new Date().toISOString(), profit: expectedProfit(validated, outcome) };
    validateEntry(settled);
    return settled;
}

export function journalSummary(entries) {
    const validated = validateJournal(entries);
    const resolved = validated.filter(entry => entry.outcome === "won" || entry.outcome === "lost");
    const risked = resolved.reduce((total, entry) => total + entry.stake, 0);
    const profit = resolved.reduce((total, entry) => total + entry.profit, 0);
    if (!Number.isFinite(risked) || !Number.isFinite(profit)) throw new Error("Journal totals exceed the supported numeric range.");
    const open = validated.filter(entry => entry.outcome === "open").length;
    return {
        count: entries.length,
        open,
        settled: entries.length - open,
        risked,
        profit,
        roi: resolved.length ? profit / risked : null,
        winRate: resolved.length ? resolved.filter(entry => entry.outcome === "won").length / resolved.length : null,
    };
}

export function readJournal(storage = localStorage) {
    const raw = storage.getItem(KEY);
    if (raw === null) return [];
    let entries;
    try {
        entries = JSON.parse(raw);
    } catch {
        throw new Error("The saved journal is not valid JSON. Its data has been preserved.");
    }
    return validateJournal(entries);
}

export function saveJournal(entries, storage = localStorage) {
    const validated = validateJournal(entries);
    storage.setItem(KEY, JSON.stringify(validated));
    return validated;
}
