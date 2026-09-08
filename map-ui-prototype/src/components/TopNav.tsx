import { IGuideMark } from './IGuideMark';
import { TopNavPlatform } from './TopNav.platform';
import { isPlatformVariant, type AppTab } from '../uiVariant';

export interface TopNavProps {
  onToggleSettings: () => void;
  onToggleHistory: () => void;
  sessionCount: number;
  tab: AppTab;
  onSetTab: (t: AppTab) => void;
}

const TABS: { id: AppTab; label: string; title: string }[] = [
  { id: 'chat', label: 'Chat', title: 'Ask anything — the map opens when an answer needs it' },
  { id: 'rs', label: 'RS-Embed Demo', title: 'Draw a region and run satellite-embedding operations on it' },
];

// The header for the rs-embed deployment (issue #20). This used to mirror the I-GUIDE platform
// chrome with non-functional placeholders — Collections / Apps / Support / a search box — which
// made the page look like the platform without behaving like it: every one of them was dead on
// click. They are gone, and the one link that DOES go somewhere replaces them.
//
// The MARK is the link back to the platform, so there is no separate text link. Only History
// and the settings gear remain on the right: the jpy badge and the account avatar were platform
// placeholders that did nothing here.
function TopNavRsEmbed(p: TopNavProps) {
  return (
    <header className="bar">
      <div className="bar-inner">
        <div className="brand">
          <a className="marklink" href="https://platform.i-guide.io" target="_blank"
             rel="noopener noreferrer" aria-label="Back to the I-GUIDE Platform"
             title="Back to the I-GUIDE Platform">
            <IGuideMark className="iglogo" />
          </a>
          <span className="brand-name">I-GUIDE AI</span>
        </div>
        {/* A demo surface, not a second app: the tabs choose what the page is SET UP for, and
            the conversation carries across both. */}
        <nav className="tabs" role="tablist" aria-label="Workspace">
          {TABS.map((t) => (
            <button key={t.id} role="tab" type="button" title={t.title}
              aria-selected={p.tab === t.id}
              className={`tab ${p.tab === t.id ? 'on' : ''}`}
              onClick={() => p.onSetTab(t.id)}>{t.label}</button>
          ))}
        </nav>
        <div className="grow" />
        <button className="navbtn" title="Past conversations" onClick={p.onToggleHistory}>
          History{p.sessionCount ? ` (${p.sessionCount})` : ''}
        </button>
        <button className="navbtn gear" title="Connection settings" onClick={p.onToggleSettings}>⚙</button>
      </div>
    </header>
  );
}

// The original prototype page is kept verbatim in TopNav.platform.tsx and selected here, so
// switching back is a build flag rather than a revert. See src/uiVariant.ts.
export function TopNav(p: TopNavProps) {
  return isPlatformVariant ? <TopNavPlatform {...p} /> : <TopNavRsEmbed {...p} />;
}
