document.addEventListener("DOMContentLoaded", () => {
    const MODEL_DISPLAY = {
        logistic_regression: "Logistic regression",
        random_forest: "Random forest",
        gradient_boosting: "Gradient boosting",
        ensemble: "Ensemble mean",
    };
    const formatDate = value => value ? new Date(`${value.slice(0, 10)}T12:00:00Z`).toLocaleDateString("en-US", {
        month: "short", day: "numeric", year: "numeric", timeZone: "UTC",
    }) : "Not recorded";
    const formatPercent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";

    async function readJSON(url, options = {}) {
        const response = await fetch(url, { ...options, signal: AbortSignal.timeout(55000) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "The server could not complete this request.");
        return data;
    }

    readJSON("/api/model-info").then(info => {
        const models = info.models || {};
        const fields = {
            fighterCount: info.fighterCount?.toLocaleString() ?? "—",
            fightCount: info.fightCount?.toLocaleString() ?? "—",
            featureCount: info.featureCount ?? "—",
            modelCount: Object.keys(models).length,
            modelNames: Object.keys(models).map(key => MODEL_DISPLAY[key] || key).join(", "),
            dateRange: info.dateRange?.start && info.dateRange?.end
                ? `${formatDate(info.dateRange.start)} – ${formatDate(info.dateRange.end)}` : "Not recorded",
            profileDateRange: info.profileDateRange?.start && info.profileDateRange?.end
                ? `${formatDate(info.profileDateRange.start)} – ${formatDate(info.profileDateRange.end)}` : "Not recorded",
            trainedAt: formatDate(info.trainedAt),
            dataThrough: formatDate(info.dateRange?.end),
        };
        document.querySelectorAll("[data-model-field]").forEach(element => {
            element.textContent = fields[element.dataset.modelField] ?? "—";
        });
        const status = document.getElementById("model-info-status");
        if (status) status.textContent = info.ready === false ? "Prediction data is unavailable. Try again shortly." : "Loaded from the model and its saved datasets.";
        const table = document.getElementById("model-evaluation");
        if (table) {
            table.innerHTML = Object.entries(models).map(([key, metrics]) => `<tr>
                <th scope="row">${esc(MODEL_DISPLAY[key] || key)}</th>
                <td>${formatPercent(metrics.accuracy)}</td>
                <td>${formatPercent(metrics.ts_accuracy)}</td>
            </tr>`).join("") || '<tr><td colspan="3">No evaluation scores were saved with this model.</td></tr>';
        }
    }).catch(() => {
        const status = document.getElementById("model-info-status");
        if (status) status.textContent = "Dataset details are unavailable. Reload to try again.";
        const table = document.getElementById("model-evaluation");
        if (table) table.innerHTML = '<tr><td colspan="3">Evaluation scores could not be loaded.</td></tr>';
    });

    const form = document.getElementById("matchup-form");
    if (!form) return;

    const f1Input = document.getElementById("fighter1");
    const f2Input = document.getElementById("fighter2");
    const ac1 = document.getElementById("autocomplete1");
    const ac2 = document.getElementById("autocomplete2");
    const matchBtn = document.getElementById("predict-btn");
    const errorEl = document.getElementById("form-error");
    const statusEl = document.getElementById("form-status");
    const retryBtn = document.getElementById("retry-fighters");
    const resultSec = document.getElementById("result-section");
    const predictionEl = document.getElementById("prediction-top");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let allFighters = [];
    let matchupRevision = 0;
    let fightersReady = false;
    let pendingRestore = "url";
    const normalizeName = value => value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").trim().replace(/\s+/g, " ").toLowerCase();

    function showError(message, input) {
        errorEl.textContent = message;
        errorEl.hidden = !message;
        if (input) {
            input.setAttribute("aria-invalid", "true");
            input.focus();
        }
    }

    function setBusy(busy) {
        form.setAttribute("aria-busy", String(busy));
        f1Input.disabled = busy;
        f2Input.disabled = busy;
        matchBtn.disabled = busy;
        document.getElementById("example-matchup").disabled = busy;
    }

    async function loadFighters() {
        setBusy(true);
        retryBtn.hidden = true;
        showError("");
        statusEl.textContent = "Loading the fighter list…";
        try {
            const data = await readJSON("/api/fighters");
            if (!Array.isArray(data.fighters) || !data.fighters.length) throw new Error("No fighters available");
            allFighters = data.fighters.filter(name => typeof name === "string");
            fightersReady = true;
            setBusy(false);
            statusEl.textContent = "";
            if (pendingRestore) {
                const origin = pendingRestore;
                pendingRestore = null;
                restoreMatchup(origin);
            }
        } catch {
            form.setAttribute("aria-busy", "false");
            f1Input.disabled = false;
            f2Input.disabled = false;
            statusEl.textContent = "";
            showError("Model lookup is unavailable. You can enter fighter names and compare prices below, or retry the lookup.");
            retryBtn.hidden = false;
        }
    }
    retryBtn.addEventListener("click", loadFighters);
    loadFighters();
    document.getElementById("example-matchup").addEventListener("click", () => {
        f1Input.value = "Alexander Volkanovski";
        f2Input.value = "Diego Lopes";
        f1Input.dispatchEvent(new Event("input", { bubbles: true }));
        f2Input.dispatchEvent(new Event("input", { bubbles: true }));
        [ac1, ac2].forEach(list => { list.replaceChildren(); list.classList.add("hidden"); });
        [f1Input, f2Input].forEach(input => input.setAttribute("aria-expanded", "false"));
        form.requestSubmit();
    });

    function setupAutocomplete(input, list) {
        let activeIndex = -1;
        function close() {
            list.replaceChildren();
            list.classList.add("hidden");
            input.setAttribute("aria-expanded", "false");
            input.removeAttribute("aria-activedescendant");
            activeIndex = -1;
        }
        function select(name) {
            input.value = name;
            input.dispatchEvent(new Event("input", { bubbles: true }));
            close();
        }
        function show() {
            close();
            const query = normalizeName(input.value);
            if (!query) return;
            const rank = name => name.startsWith(query) ? 0 : name.includes(` ${query}`) ? 1 : 2;
            const matches = allFighters.map(name => ({name, normalized:normalizeName(name)}))
                .filter(fighter => fighter.normalized.includes(query))
                .sort((first, second) => rank(first.normalized) - rank(second.normalized) || first.name.localeCompare(second.name))
                .slice(0, 10);
            list.classList.remove("hidden");
            input.setAttribute("aria-expanded", "true");
            if (!matches.length) {
                const item = document.createElement("li");
                item.setAttribute("role", "presentation");
                item.classList.add("autocomplete-empty");
                item.textContent = "No matching fighters. Try another name.";
                list.appendChild(item);
                return;
            }
            matches.forEach(({name}, index) => {
                const item = document.createElement("li");
                item.id = `${list.id}-option-${index}`;
                item.setAttribute("role", "option");
                item.setAttribute("aria-selected", "false");
                item.textContent = name;
                item.addEventListener("mousedown", event => event.preventDefault());
                item.addEventListener("click", () => select(name));
                list.appendChild(item);
            });
        }
        input.addEventListener("input", () => {
            matchupRevision += 1;
            showError("");
            statusEl.textContent = "";
            input.removeAttribute("aria-invalid");
            resultSec.classList.add("hidden");
            predictionEl.classList.add("idle");
            window.dispatchEvent(new CustomEvent("versus:matchup-change"));
            show();
        });
        input.addEventListener("focus", show);
        input.addEventListener("keydown", event => {
            if (event.key === "Escape" || event.key === "Tab") return close();
            if ((event.key === "ArrowDown" || event.key === "ArrowUp") && input.getAttribute("aria-expanded") !== "true") show();
            const items = [...list.children].filter(item => item.getAttribute("role") === "option");
            if (!items.length) return;
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                event.preventDefault();
                activeIndex = Math.max(0, Math.min(items.length - 1, activeIndex + (event.key === "ArrowDown" ? 1 : -1)));
                items.forEach((item, index) => {
                    item.classList.toggle("active", index === activeIndex);
                    item.setAttribute("aria-selected", String(index === activeIndex));
                });
                input.setAttribute("aria-activedescendant", items[activeIndex].id);
                items[activeIndex].scrollIntoView({block:"nearest", inline:"nearest"});
            } else if (event.key === "Enter" && activeIndex >= 0) {
                event.preventDefault();
                select(items[activeIndex].textContent);
            }
        });
        input.addEventListener("blur", close);
    }
    setupAutocomplete(f1Input, ac1);
    setupAutocomplete(f2Input, ac2);

    function closeSuggestions() {
        [ac1, ac2].forEach(list => { list.replaceChildren(); list.classList.add("hidden"); });
        [f1Input, f2Input].forEach(input => {
            input.setAttribute("aria-expanded", "false");
            input.removeAttribute("aria-activedescendant");
        });
    }

    function startResearch(origin = "form") {
        showError("");
        if (!fightersReady) return;
        [f1Input, f2Input].forEach(input => input.removeAttribute("aria-invalid"));
        for (const input of [f1Input, f2Input]) {
            const name = allFighters.find(fighter => normalizeName(fighter) === normalizeName(input.value));
            if (!name) return showError("Choose a fighter from the suggestions.", input);
            input.value = name;
        }
        if (f1Input.value === f2Input.value) return showError("Choose two different fighters.", f2Input);
        const fighter1 = f1Input.value;
        const fighter2 = f2Input.value;
        resultSec.classList.add("hidden");
        predictionEl.classList.add("idle");
        closeSuggestions();
        window.dispatchEvent(new CustomEvent("versus:research-start", { detail: { fighter1, fighter2, origin } }));
        statusEl.textContent = "";
    }

    function restoreMatchup(origin) {
        const params = new URL(window.location.href).searchParams;
        showError("");
        statusEl.textContent = "";
        [f1Input, f2Input].forEach(input => input.removeAttribute("aria-invalid"));
        f1Input.value = params.get("fighter1") || "";
        f2Input.value = params.get("fighter2") || "";
        closeSuggestions();
        window.dispatchEvent(new CustomEvent("versus:matchup-change", { detail: { preserveURL: true } }));
        if (f1Input.value && f2Input.value) startResearch(origin);
        else if (f1Input.value || f2Input.value) showError("Choose both fighters to open this matchup.");
    }

    window.addEventListener("versus:restore-matchup", () => {
        if (fightersReady) restoreMatchup("history");
        else pendingRestore = "history";
    });
    form.addEventListener("submit", event => {
        event.preventDefault();
        startResearch();
    });

    function esc(str) {
        const d = document.createElement("div");
        d.textContent = str;
        return d.innerHTML;
    }

    function fmt(val, suffix) {
        if (val === null || val === undefined || (typeof val === "number" && !Number.isFinite(val))) return "\u2014";
        if (typeof val === "number") val = Number(val.toFixed(1));
        return suffix ? esc(val + suffix) : esc(String(val));
    }

    // Convert win probability to American moneyline odds.
    function toMoneyline(prob) {
        if (prob >= 0.999) return "-∞";
        if (prob <= 0.001) return "+∞";
        if (prob >= 0.5) return "\u2212" + Math.round((100 * prob) / (1 - prob));
        return "+" + Math.round((100 * (1 - prob)) / prob);
    }

    // ---- Render Results ----
    function renderResults(data) {
        const { fighter1, fighter2, winner, f1Prob, f2Prob, f1, f2, analysis } = data;
        const p1 = (f1Prob * 100).toFixed(1);
        const p2 = (f2Prob * 100).toFixed(1);

        // Probability bar
        document.getElementById("prob-f1-name").textContent = fighter1;
        document.getElementById("prob-f2-name").textContent = fighter2;
        document.getElementById("prob-f1-val").textContent = p1 + "%";
        document.getElementById("prob-f2-val").textContent = p2 + "%";

        // Moneyline odds
        const oddsEl1 = document.getElementById("odds-f1");
        const oddsEl2 = document.getElementById("odds-f2");
        if (oddsEl1) oddsEl1.textContent = toMoneyline(f1Prob);
        if (oddsEl2) oddsEl2.textContent = toMoneyline(f2Prob);

        const f1Fill = document.getElementById("prob-f1-fill");
        const f2Fill = document.getElementById("prob-f2-fill");
        f1Fill.style.width = "0%";
        f2Fill.style.width = "0%";

        document.getElementById("winner-name").textContent = winner ?? "neither fighter";
        const arrowL = document.getElementById("winner-arrow");
        const arrowR = document.getElementById("winner-arrow-r");
        arrowL.classList.toggle("active", winner === fighter1);
        arrowR.classList.toggle("active", winner === fighter2);

        // Fighter profiles
        renderProfile("f1", fighter1, f1);
        renderProfile("f2", fighter2, f2);

        // Comparison bars
        renderComparison(fighter1, fighter2, f1, f2);

        // Win methods
        renderMethods(f1, f2);

        // Analysis sections
        if (analysis) {
            renderKeyFactors(analysis.keyFactors, fighter1, fighter2);
            renderCategoryEdges(analysis.categories, fighter1, fighter2);
            renderModelBreakdown(analysis.modelBreakdown, fighter1, fighter2);
            renderHistorical(analysis.historicalMatchups, fighter1, fighter2);
        }

        // Reveal result section and animate bar fill
        document.getElementById("model-panel").hidden = false;
        document.getElementById("model-details").hidden = false;
        document.getElementById("model-status").hidden = true;
        predictionEl.classList.remove("idle");
        resultSec.classList.remove("hidden");
        resultSec.querySelectorAll(".reveal").forEach(element => element.classList.add("visible"));
        setTimeout(() => {
            f1Fill.style.width = p1 + "%";
            f2Fill.style.width = p2 + "%";
        }, 80);

        window.dispatchEvent(new CustomEvent("versus:prediction", { detail: data }));
    }

    // ---- Fighter Profile ----
    function renderProfile(prefix, name, stats) {
        document.getElementById(`profile-${prefix}-name`).textContent = name;
        document.getElementById(`profile-${prefix}-record`).textContent = stats.record ?? "—";
        document.getElementById(`profile-${prefix}-date`).textContent = stats.available
            ? `Profile before ${formatDate(stats.asOf)}` : "Career profile unavailable";
        document.getElementById(`feature-${prefix}-date`).textContent = `Last bout in model data: ${formatDate(stats.featureAsOf)}`;
        document.getElementById(`profile-${prefix}-stance`).textContent = stats.stance || "Unknown";
        document.getElementById(`profile-${prefix}-age`).textContent = Number.isFinite(stats.age) ? `Age ${Math.floor(stats.age)}` : "";

        const grid = document.getElementById(`stats-${prefix}`);
        grid.innerHTML = "";

        const items = [
            ["Height", stats.height],
            ["Reach", stats.reach],
            ["Weight", stats.weight ? stats.weight + " lbs" : null],
            ["Win Streak", stats.currentWinStreak],
            ["Sig. Str/Min", fmt(stats.avgSigStrLanded)],
            ["Str. Acc.", formatPercent(stats.avgSigStrPct)],
            ["TD Avg", fmt(stats.avgTDLanded)],
            ["TD Acc.", formatPercent(stats.avgTDPct)],
            ["Sub Avg", fmt(stats.avgSubAtt)],
            ["Title Bouts", fmt(stats.titleBouts)],
            ["Total Rnds", fmt(stats.totalRounds)],
            ["Best Streak", fmt(stats.longestWinStreak)],
        ];

        items.forEach(([label, value]) => {
            const div = document.createElement("div");
            div.className = "stat-item";
            div.innerHTML = `<span class="stat-label">${esc(label)}</span><span class="stat-value">${esc(value ?? "\u2014")}</span>`;
            grid.appendChild(div);
        });
    }

    // ---- Head-to-Head Comparison ----
    function renderComparison(name1, name2, f1, f2) {
        const rows = document.getElementById("comparison-rows");
        rows.innerHTML = "";

        const metrics = [
            ["Wins", f1.wins, f2.wins],
            ["Losses", f1.losses, f2.losses, true],
            ["Sig. Str/Min", f1.avgSigStrLanded, f2.avgSigStrLanded],
            ["Str. Accuracy", f1.avgSigStrPct, f2.avgSigStrPct, false, true],
            ["TD Average", f1.avgTDLanded, f2.avgTDLanded],
            ["TD Accuracy", f1.avgTDPct, f2.avgTDPct, false, true],
            ["Sub Attempts", f1.avgSubAtt, f2.avgSubAtt],
            ["Win Streak", f1.currentWinStreak, f2.currentWinStreak],
        ];

        metrics.forEach(([label, v1, v2, invertAdvantage, percentage]) => {
            const val1 = v1 ?? 0;
            const val2 = v2 ?? 0;
            const total = val1 + val2 || 1;
            const pct1 = (val1 / total) * 100;
            const pct2 = (val2 / total) * 100;
            const complete = Number.isFinite(v1) && Number.isFinite(v2);
            const leftLeads = !complete ? false : invertAdvantage ? val1 < val2 : val1 > val2;
            const rightLeads = !complete ? false : invertAdvantage ? val2 < val1 : val2 > val1;

            const row = document.createElement("div");
            row.className = "cmp-row";
            row.innerHTML = `
                <div class="cmp-val left ${leftLeads ? 'advantage' : ''}">${percentage ? formatPercent(v1) : fmt(v1)}</div>
                <div class="cmp-center">
                    <div class="cmp-label">${esc(label)}</div>
                    <div class="cmp-bar-wrap">
                        <div class="cmp-bar-left ${leftLeads ? 'lead' : ''}" style="width:${complete ? pct1 : 0}%"></div>
                        <div class="cmp-bar-right ${rightLeads ? 'lead' : ''}" style="width:${complete ? pct2 : 0}%"></div>
                    </div>
                </div>
                <div class="cmp-val right ${rightLeads ? 'advantage' : ''}">${percentage ? formatPercent(v2) : fmt(v2)}</div>
            `;
            rows.appendChild(row);
        });
    }

    // ---- Win Methods ----
    function renderMethods(f1, f2) {
        const methods = [
            ["KO/TKO", f1.koWins, f2.koWins],
            ["Submission", f1.subWins, f2.subWins],
            ["Decision", f1.decWins, f2.decWins],
        ];
        const maxVal = Math.max(...methods.flatMap(([, a, b]) => [a || 0, b || 0]), 1);

        const colF1 = document.getElementById("method-f1");
        const colF2 = document.getElementById("method-f2");
        const labels = document.getElementById("method-labels");
        colF1.innerHTML = "";
        colF2.innerHTML = "";
        labels.innerHTML = "";

        methods.forEach(([label, v1, v2]) => {
            const w1 = Math.max(((v1 || 0) / maxVal) * 100, 3);
            const w2 = Math.max(((v2 || 0) / maxVal) * 100, 3);
            labels.innerHTML += `<div class="method-row-label">${esc(label)}</div>`;
            colF1.innerHTML += `
                <div class="method-bar-wrap left">
                    <div class="method-bar red-bar" style="width:${w1}%">
                        <span class="method-count">${fmt(v1)}</span>
                    </div>
                </div>`;
            colF2.innerHTML += `
                <div class="method-bar-wrap right">
                    <div class="method-bar blue-bar" style="width:${w2}%">
                        <span class="method-count">${fmt(v2)}</span>
                    </div>
                </div>`;
        });
    }

    // ---- Key Factors (NEW) ----
    const CATEGORY_COLORS = {
        striking: "#56514a",
        grappling: "#56514a",
        physical: "#56514a",
        experience: "#56514a",
        other: "#56514a",
    };

    function renderKeyFactors(factors, f1Name, f2Name) {
        const container = document.getElementById("key-factors");
        container.innerHTML = "";
        if (!factors || !factors.length) return;

        const maxImpact = Math.max(...factors.map(f => f.impact));

        factors.forEach(factor => {
            const isF1 = factor.advantage === f1Name;
            const barPct = Math.max((factor.impact / maxImpact) * 100, 8);
            const catColor = CATEGORY_COLORS[factor.category] || CATEGORY_COLORS.other;

            const row = document.createElement("div");
            row.className = "factor-row";
            row.innerHTML = `
                <div class="factor-top">
                    <span class="factor-name">${esc(factor.factor)}</span>
                    <span class="factor-cat" style="background: ${catColor}20; color: ${catColor}; border-color: ${catColor}40">${esc(factor.category)}</span>
                </div>
                <div class="factor-bottom">
                    <span class="factor-val ${isF1 ? 'red-text' : ''}">${fmt(factor.f1Value)}</span>
                    <div class="factor-bar-wrap">
                        <div class="factor-bar ${isF1 ? 'red-fill' : 'blue-fill'}" style="width:${barPct}%"></div>
                    </div>
                    <span class="factor-val ${!isF1 ? 'blue-text' : ''}">${fmt(factor.f2Value)}</span>
                </div>
                <div class="factor-adv ${isF1 ? 'red-text' : 'blue-text'}">
                    ${isF1 ? '\u25C0' : '\u25B6'} Favors ${esc(factor.advantage)}
                    ${factor.imputed ? '<span class="imputed-note">Missing input; training median used</span>' : ''}
                </div>
            `;
            container.appendChild(row);
        });
    }

    // ---- Category Edge Analysis (NEW) ----
    function renderCategoryEdges(categories, f1Name, f2Name) {
        const container = document.getElementById("category-edges");
        container.innerHTML = "";
        if (!categories) return;

        document.getElementById("edge-f1-name").textContent = f1Name;
        document.getElementById("edge-f2-name").textContent = f2Name;

        const order = ["striking", "grappling", "physical", "experience"];

        order.forEach(key => {
            const cat = categories[key];
            if (!cat) return;

            const score = cat.score;
            const f1Pct = score;
            const f2Pct = 100 - score;
            const catColor = CATEGORY_COLORS[key] || CATEGORY_COLORS.other;

            const row = document.createElement("div");
            row.className = "edge-row";
            row.innerHTML = `
                <div class="edge-label">
                    <span class="edge-dot" style="background: ${catColor}"></span>
                    <span class="edge-name">${esc(cat.label)}</span>
                </div>
                <div class="edge-bar-container">
                    <div class="edge-bar-track">
                        <div class="edge-fill-left" style="width:${f1Pct}%"></div>
                        <div class="edge-fill-right" style="width:${f2Pct}%"></div>
                    </div>
                    <div class="edge-midline"></div>
                </div>
                <div class="edge-score ${score > 55 ? 'red-text' : score < 45 ? 'blue-text' : ''}">${score}</div>
            `;
            container.appendChild(row);
        });
    }

    // ---- Model Breakdown (NEW) ----
    function renderModelBreakdown(breakdown, f1Name, f2Name) {
        const container = document.getElementById("model-breakdown");
        container.innerHTML = "";
        if (!breakdown) return;

        document.getElementById("models-f1-name").textContent = f1Name;
        document.getElementById("models-f2-name").textContent = f2Name;

        const order = [...Object.keys(breakdown).filter(key => key !== "ensemble"), ...("ensemble" in breakdown ? ["ensemble"] : [])];

        order.forEach(key => {
            const model = breakdown[key];
            if (!model) return;

            const f1Pct = (model.f1Prob * 100).toFixed(1);
            const f2Pct = (model.f2Prob * 100).toFixed(1);
            const isEnsemble = key === "ensemble";


            const row = document.createElement("div");
            row.className = `model-row ${isEnsemble ? "model-ensemble" : ""}`;
            row.innerHTML = `
                <div class="model-info">
                    <span class="model-name">${esc(MODEL_DISPLAY[key] || key)}</span>

                </div>
                <div class="model-bar-wrap">
                    <div class="model-bar">
                        <div class="model-fill red-fill" style="width:${f1Pct}%"></div>
                        <div class="model-fill blue-fill" style="width:${f2Pct}%"></div>
                    </div>
                </div>
                <div class="model-probs">
                    <span class="model-prob red-text">${f1Pct}%</span>
                    <span class="model-prob blue-text">${f2Pct}%</span>
                </div>
            `;
            container.appendChild(row);
        });
    }

    // ---- Historical Matchups (NEW) ----
    function renderHistorical(matchups, f1Name, f2Name) {
        const card = document.getElementById("history-card");
        const container = document.getElementById("historical-matchups");
        container.innerHTML = "";

        if (!matchups || !matchups.length) {
            card.classList.add("hidden");
            return;
        }

        card.classList.remove("hidden");

        matchups.forEach(m => {
            const dateStr = esc(m.date || "Unknown date");
            const methodParts = [m.method];
            if (m.round) methodParts.push(`Round ${m.round}`);
            if (m.time) methodParts.push(m.time);

            const isF1Winner = m.winner === f1Name;
            const isF2Winner = m.winner === f2Name;

            const row = document.createElement("div");
            row.className = "history-row";
            row.innerHTML = `
                <div class="history-date">${dateStr}</div>
                <div class="history-result">
                    <span class="history-winner ${isF1Winner ? 'red-text' : isF2Winner ? 'blue-text' : ''}">${esc(m.winner)}</span>
                    <span class="history-method">${esc(methodParts.filter(Boolean).join(" \u00B7 "))}</span>
                </div>
            `;
            container.appendChild(row);
        });
    }
});
