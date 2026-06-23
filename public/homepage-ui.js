(() => {
    const form = document.getElementById('home-matchup-form');
    const firstInput = document.getElementById('home-fighter-one');
    const secondInput = document.getElementById('home-fighter-two');
    const submitButton = document.getElementById('home-search-submit');
    const searchStatus = document.getElementById('home-search-status');
    const searchRecovery = document.getElementById('home-search-recovery');
    const recordsState = document.getElementById('home-records-state');
    const recordsTable = document.getElementById('home-records-table');
    const fightRows = document.getElementById('home-fight-rows');
    let fighters = [];
    let fighterIndex = new Map();

    const normalize = value => value.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').trim().toLowerCase();
    const makeElement = (tag, className, text) => {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== undefined) element.textContent = text;
        return element;
    };
    const researchUrl = (first, second) => `/research?${new URLSearchParams({ fighter1: first, fighter2: second, view: 'report' })}`;
    const readJson = async url => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 55000);
        try {
            const response = await fetch(url, { signal: controller.signal });
            if (!response.ok) throw new Error('Request unavailable');
            return await response.json();
        } finally {
            clearTimeout(timeout);
        }
    };

    function setSearchStatus(message, error = false) {
        searchStatus.textContent = message;
        searchStatus.classList.toggle('is-error', error);
    }

    function autocomplete(input, list) {
        let matches = [];
        let activeIndex = -1;
        const close = () => {
            list.hidden = true;
            input.setAttribute('aria-expanded', 'false');
            input.removeAttribute('aria-activedescendant');
            activeIndex = -1;
        };
        const select = index => {
            if (!matches[index]) return;
            input.value = matches[index];
            input.removeAttribute('aria-invalid');
            close();
            input.focus();
        };
        const activate = index => {
            activeIndex = index;
            [...list.querySelectorAll('[role="option"]')].forEach((option, optionIndex) => {
                option.setAttribute('aria-selected', String(optionIndex === index));
                if (optionIndex === index) {
                    input.setAttribute('aria-activedescendant', option.id);
                    option.scrollIntoView({ block: 'nearest' });
                }
            });
        };
        const show = () => {
            const query = normalize(input.value);
            if (!query || !fighters.length) {
                matches = [];
                return close();
            }
            matches = fighters.filter(name => normalize(name).includes(query))
                .sort((a, b) => Number(!normalize(a).startsWith(query)) - Number(!normalize(b).startsWith(query)) || a.localeCompare(b))
                .slice(0, 8);
            list.replaceChildren();
            activeIndex = -1;
            input.removeAttribute('aria-activedescendant');
            if (!matches.length) {
                const empty = makeElement('li', 'home-no-match', 'No matching fighters. Try another name.');
                empty.setAttribute('role', 'presentation');
                list.append(empty);
            }
            matches.forEach((name, index) => {
                const option = makeElement('li', '', name);
                option.id = `${list.id}-${index}`;
                option.setAttribute('role', 'option');
                option.setAttribute('aria-selected', 'false');
                option.addEventListener('pointerdown', event => event.preventDefault());
                option.addEventListener('click', () => select(index));
                list.append(option);
            });
            list.hidden = false;
            input.setAttribute('aria-expanded', 'true');
        };
        input.addEventListener('input', () => {
            input.removeAttribute('aria-invalid');
            if (fighters.length) setSearchStatus('Search fighters in the recorded dataset.');
            show();
        });
        input.addEventListener('focus', show);
        input.addEventListener('blur', close);
        input.addEventListener('keydown', event => {
            if (event.key === 'Escape') return close();
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault();
                if (list.hidden) show();
                if (list.hidden || !matches.length) return;
                const nextIndex = event.key === 'ArrowDown'
                    ? (activeIndex + 1) % matches.length
                    : activeIndex <= 0 ? matches.length - 1 : activeIndex - 1;
                activate(nextIndex);
            }
            if (event.key === 'Enter' && !list.hidden && activeIndex >= 0) {
                event.preventDefault();
                select(activeIndex);
            }
            if (event.key === 'Tab') close();
        });
        return close;
    }

    const closeFirst = autocomplete(firstInput, document.getElementById('home-suggestions-one'));
    const closeSecond = autocomplete(secondInput, document.getElementById('home-suggestions-two'));

    async function loadFighters() {
        searchRecovery.hidden = true;
        setSearchStatus('Loading fighter search…');
        try {
            const data = await readJson('/api/fighters');
            if (!Array.isArray(data.fighters)) throw new Error('No fighter list');
            fighters = [...new Set(data.fighters.filter(name => typeof name === 'string' && name.trim()))];
            if (!fighters.length) throw new Error('Empty fighter list');
            fighterIndex = new Map(fighters.map(name => [normalize(name), name]));
            setSearchStatus('Search fighters in the recorded dataset.');
        } catch {
            setSearchStatus('Suggestions are unavailable. Enter both fighters’ full names to continue.', true);
            searchRecovery.hidden = false;
        } finally {
            firstInput.disabled = false;
            secondInput.disabled = false;
            submitButton.disabled = false;
        }
    }

    form.addEventListener('submit', event => {
        event.preventDefault();
        closeFirst();
        closeSecond();
        const first = fighterIndex.get(normalize(firstInput.value)) ?? (!fighters.length ? firstInput.value.trim() : '');
        const second = fighterIndex.get(normalize(secondInput.value)) ?? (!fighters.length ? secondInput.value.trim() : '');
        if (!first || !second) {
            const input = !first ? firstInput : secondInput;
            input.setAttribute('aria-invalid', 'true');
            setSearchStatus('Choose a fighter from the suggestions for each field.', true);
            input.focus();
            return;
        }
        if (normalize(first) === normalize(second)) {
            secondInput.setAttribute('aria-invalid', 'true');
            setSearchStatus('Choose two different fighters.', true);
            secondInput.focus();
            return;
        }
        window.location.assign(researchUrl(first, second));
    });
    document.getElementById('home-search-retry').addEventListener('click', loadFighters);

    function dateValue(value) {
        if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}/.test(value)) return null;
        const iso = value.slice(0, 10);
        const date = new Date(`${iso}T12:00:00Z`);
        if (Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== iso) return null;
        return { iso, label: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }) };
    }

    function renderFight(fight) {
        const row = makeElement('tr');
        const dateCell = makeElement('td', 'home-fight-date');
        const date = dateValue(fight.date);
        if (date) {
            const time = makeElement('time', '', date.label);
            time.dateTime = date.iso;
            dateCell.append(time);
        } else dateCell.textContent = 'Date not recorded';
        const matchupCell = makeElement('td');
        const matchup = makeElement('a', 'home-fight-link');
        matchup.href = researchUrl(fight.firstFighter, fight.secondFighter);
        matchup.setAttribute('aria-label', `${fight.firstFighter} vs ${fight.secondFighter}`);
        matchup.append(makeElement('span', '', fight.firstFighter), makeElement('span', 'home-versus', 'vs'), makeElement('span', '', fight.secondFighter));
        matchupCell.append(matchup);
        if (typeof fight.weightClass === 'string' && fight.weightClass) matchupCell.append(makeElement('span', 'home-fight-division', fight.weightClass));
        const resultCell = makeElement('td');
        const result = fight.result === 'draw' ? 'Draw' : fight.result === 'nc' ? 'No contest'
            : typeof fight.winner === 'string' && fight.winner ? `${fight.winner} won` : 'Result not recorded';
        resultCell.append(makeElement('span', 'home-fight-outcome', result));
        if (typeof fight.method === 'string' && fight.method) resultCell.append(makeElement('span', 'home-fight-method', fight.method));
        const actionCell = makeElement('td');
        const open = makeElement('a', 'home-fight-open', 'Research');
        open.href = matchup.href;
        open.setAttribute('aria-label', `Research ${fight.firstFighter} versus ${fight.secondFighter}`);
        const arrow = makeElement('span', '', '→');
        arrow.setAttribute('aria-hidden', 'true');
        open.append(arrow);
        actionCell.append(open);
        row.append(dateCell, matchupCell, resultCell, actionCell);
        return row;
    }

    function renderSource(source) {
        const note = document.getElementById('home-records-source');
        note.textContent = 'Completed fights in the research dataset.';
        if (!source || typeof source.sourceUrl !== 'string') return;
        let url;
        try { url = new URL(source.sourceUrl); } catch { return; }
        if (!['http:', 'https:'].includes(url.protocol)) return;
        const link = makeElement('a', '', typeof source.source === 'string' && source.source ? source.source : 'Fight data source');
        link.href = url.href;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        note.append(' Source: ', link, '.');
    }

    async function loadRecords() {
        recordsState.hidden = false;
        recordsState.replaceChildren(makeElement('p', '', 'Loading fight records…'));
        try {
            const data = await readJson('/api/recent-fights');
            if (!Array.isArray(data.fights)) throw new Error('No fight records');
            const recordedFights = data.fights.filter(fight => fight && typeof fight.firstFighter === 'string' && fight.firstFighter.trim() && typeof fight.secondFighter === 'string' && fight.secondFighter.trim()).slice(0, 8);
            if (!recordedFights.length) throw new Error('Empty fight records');
            fightRows.replaceChildren(...recordedFights.map(renderFight));
            recordsTable.hidden = false;
            recordsState.hidden = true;
            const through = dateValue(data.dataThrough);
            document.getElementById('home-coverage').textContent = through ? `Records through ${through.label}` : 'Historical results from the research dataset.';
            renderSource(data.source);
        } catch {
            recordsTable.hidden = true;
            recordsState.replaceChildren(makeElement('p', '', 'Fight records are unavailable right now. You can still open the research workspace.'));
            const retry = makeElement('button', '', 'Try again');
            retry.type = 'button';
            retry.addEventListener('click', loadRecords);
            const research = makeElement('a', '', 'Open research');
            research.href = '/research';
            recordsState.append(retry, research);
        }
    }

    loadFighters();
    loadRecords();
})();
