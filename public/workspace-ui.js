const navigation = document.getElementById('workspace-navigation');

if (navigation) {
    const tabs = [...document.querySelectorAll('[data-workspace-tab]')];
    const panels = [...document.querySelectorAll('[data-workspace-panel]')];
    const views = tabs.map(tab => tab.dataset.workspaceTab);
    const $ = id => document.getElementById(id);
    let selected = null;
    let evidenceReady = false;
    let researchFailed = false;
    let editingMatchup = false;
    let currentView = readView();

    function hashTarget() {
        return document.getElementById(window.location.hash.slice(1));
    }

    function readView() {
        const linkedView = hashTarget()?.closest('[data-workspace-panel]')?.dataset.workspacePanel;
        const requested = linkedView || new URL(window.location.href).searchParams.get('view');
        return views.includes(requested) ? requested : 'report';
    }

    function matchupURL(keepHash = false) {
        const url = new URL(window.location.pathname, window.location.origin);
        if (selected) {
            url.searchParams.set('fighter1', selected.fighter1);
            url.searchParams.set('fighter2', selected.fighter2);
        }
        url.searchParams.set('view', currentView);
        if (keepHash) url.hash = window.location.hash;
        return url;
    }

    function writeURL(mode = 'replace', keepHash = false) {
        const url = matchupURL(keepHash);
        if (url.href !== window.location.href) window.history[`${mode}State`](null, '', url);
    }

    function renderEmpty() {
        const needsReport = currentView === 'report' && !selected;
        const needsEstimate = currentView === 'estimate' && !evidenceReady;
        $('workspace-empty').hidden = !needsReport && !needsEstimate;
        const title = $('workspace-empty-title');
        const description = $('workspace-empty-description');
        if (needsEstimate && selected) {
            if (title) title.textContent = researchFailed ? 'The report needs to load first' : 'Loading the matchup';
            if (description) description.textContent = researchFailed
                ? 'Open Report and retry loading the fight records.'
                : 'The estimate setup will appear when the fight records are ready.';
        } else {
            if (title) title.textContent = 'Choose a matchup to get started';
            if (description) description.textContent = currentView === 'estimate'
                ? 'Compare two fighters above, then set the date, division and rounds.'
                : 'Search for two fighters above or open the example matchup.';
        }
        $('workspace-selection').textContent = selected ? `${selected.fighter1} vs ${selected.fighter2}` : 'No matchup selected';
        $('share-matchup').disabled = !selected;
    }

    function clearShareStatus() {
        $('share-status').textContent = '';
        if ($('share-link')) $('share-link').hidden = true;
    }

    function setView(view, {history = false, keepHash = false, scroll = false} = {}) {
        const nextView = views.includes(view) ? view : 'report';
        const resetScroll = scroll && nextView !== currentView
            && panels.find(panel => panel.dataset.workspacePanel === currentView).getBoundingClientRect().top < navigation.getBoundingClientRect().bottom;
        currentView = nextView;
        for (const tab of tabs) {
            const active = tab.dataset.workspaceTab === currentView;
            tab.setAttribute('aria-selected', String(active));
            tab.tabIndex = active ? 0 : -1;
        }
        for (const panel of panels) panel.hidden = panel.dataset.workspacePanel !== currentView;
        renderEmpty();
        clearShareStatus();
        if (history) writeURL(editingMatchup ? 'replace' : history, keepHash);
        if (resetScroll) panels.find(panel => panel.dataset.workspacePanel === currentView).scrollIntoView({block:'start', behavior:'instant'});
    }

    function revealHash() {
        const target = hashTarget();
        const panel = target?.closest('[data-workspace-panel]');
        if (!target || !panel) return;
        setView(panel.dataset.workspacePanel);
        for (let parent = target; parent && parent !== panel; parent = parent.parentElement) {
            if (parent.tagName === 'DETAILS') parent.open = true;
        }
        if (!target.hidden) target.scrollIntoView({block:'start', behavior:'instant'});
    }

    tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => setView(tab.dataset.workspaceTab, {history:'push', scroll:true}));
        tab.addEventListener('keydown', event => {
            const next = {ArrowRight:(index + 1) % tabs.length, ArrowLeft:(index + tabs.length - 1) % tabs.length, Home:0, End:tabs.length - 1}[event.key];
            if (next === undefined) return;
            event.preventDefault();
            tabs[next].focus();
            setView(tabs[next].dataset.workspaceTab, {history:'push', scroll:true});
        });
    });

    window.addEventListener('versus:matchup-change', event => {
        const hadMatchup = Boolean(selected);
        selected = null;
        evidenceReady = false;
        researchFailed = false;
        renderEmpty();
        clearShareStatus();
        if (event.detail?.preserveURL) editingMatchup = false;
        else {
            if (hadMatchup) editingMatchup = true;
            writeURL(hadMatchup ? 'push' : 'replace');
        }
    });
    window.addEventListener('versus:research-start', event => {
        selected = {fighter1:event.detail.fighter1, fighter2:event.detail.fighter2};
        evidenceReady = false;
        researchFailed = false;
        const restoring = ['url', 'history'].includes(event.detail.origin);
        const history = restoring || editingMatchup ? 'replace' : 'push';
        editingMatchup = false;
        setView(restoring ? readView() : 'report', {history, keepHash:restoring});
    });
    window.addEventListener('versus:research-loading', event => {
        if (!selected || event.detail.fighter1 !== selected.fighter1 || event.detail.fighter2 !== selected.fighter2) return;
        researchFailed = false;
        renderEmpty();
    });
    window.addEventListener('versus:evidence', event => {
        if (!selected || event.detail.fighters?.[0]?.name !== selected.fighter1 || event.detail.fighters?.[1]?.name !== selected.fighter2) return;
        evidenceReady = true;
        researchFailed = false;
        renderEmpty();
        revealHash();
    });
    window.addEventListener('versus:research-error', event => {
        if (!selected || event.detail.fighter1 !== selected.fighter1 || event.detail.fighter2 !== selected.fighter2) return;
        researchFailed = true;
        renderEmpty();
    });
    window.addEventListener('hashchange', () => {
        revealHash();
        if (selected) writeURL('replace', true);
    });
    window.addEventListener('popstate', () => {
        editingMatchup = false;
        const params = new URL(window.location.href).searchParams;
        setView(readView());
        if (selected?.fighter1 === params.get('fighter1') && selected?.fighter2 === params.get('fighter2')) revealHash();
        else window.dispatchEvent(new CustomEvent('versus:restore-matchup'));
    });

    $('share-matchup').addEventListener('click', async () => {
        if (!selected) return;
        const url = matchupURL().href;
        clearShareStatus();
        try {
            await navigator.clipboard.writeText(url);
            if (!selected || url !== matchupURL().href) return;
            $('share-status').textContent = 'Matchup link copied.';
        } catch {
            if (!selected || url !== matchupURL().href) return;
            let input = $('share-link');
            if (!input) {
                input = document.createElement('input');
                input.id = 'share-link';
                input.type = 'url';
                input.readOnly = true;
                input.setAttribute('aria-label', 'Matchup link');
                $('share-status').after(input);
            }
            input.value = url;
            input.hidden = false;
            input.focus();
            input.select();
            $('share-status').textContent = 'Copy the selected link.';
        }
    });

    setView(currentView);
}
