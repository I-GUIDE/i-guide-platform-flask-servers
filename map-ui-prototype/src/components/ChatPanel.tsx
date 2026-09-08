import { useEffect, useRef, useState } from 'react';
import type { LayerArtifact } from '../contracts';
import type { FileRecord, ModelCatalogue, TraceLine } from '../agentClient';
import { SUGGESTIONS } from '../agentBrain';
import type { AppTab } from '../uiVariant';
import { groupedSources, sourceHref, type SourceGroup } from '../answerFormat';

export interface ChatMessage {
  role: 'user' | 'agent';
  text?: string;
  html?: string;
  trace?: TraceLine[];
  artifacts?: FileRecord[];
  layers?: { id: string; label: string; source: string }[];
  response?: any;
  streaming?: boolean;
}

export type Mode = 'live' | 'local';
export interface AgentCfg { endpoint: string; uploadEndpoint: string; apiKey: string;
                            model?: string; provider?: string; reasoningEffort?: string;
                            codePeer?: string; codePeerModel?: string; orchestration?: string }

const RS_ACTIONS = [
  { label: 'Embed',   prompt: 'Embed this drawn region with the gse model for June–September 2022 and put the embedding on the map.' },
  { label: 'Segment', prompt: 'Segment this drawn region into 6 look-alike zones from its satellite embedding and show it on the map.' },
  { label: 'Change',  prompt: 'How much did this drawn region change across 2018, 2020, 2022 and 2024 according to its satellite embeddings?' },
  { label: 'Predict', prompt: 'Run the available pretrained heads on this drawn region and report the predictions with their validation scores.' },
];

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  tab: AppTab;
  hasRegion: boolean;
  mapVisible: boolean;
  models: ModelCatalogue | null;
  layers: LayerArtifact[];
  mode: Mode;
  cfg: AgentCfg;
  spatial: boolean;
  showSettings: boolean;
  resolveUrl: (u: string) => string;
  onSend: (text: string) => void;
  onStop: () => void;
  onClearRegion: () => void;
  onToggleMap: () => void;
  onUpload: (files: File[]) => void;
  onSetMode: (m: Mode) => void;
  onSetCfg: (c: AgentCfg) => void;
  onSetSpatial: (v: boolean) => void;
  onToggleSettings: () => void;
}

const GROUP_LABEL: Record<SourceGroup, string> = {
  internal: 'I-GUIDE knowledge base', external: 'External open-data catalogs', web: 'Open web',
};
const isImg = (f: FileRecord) => f.kind === 'image' || /\.(png|jpe?g|gif|webp|bmp|avif)$/i.test(f.filename || f.download_url || '');

function Sources({ response }: { response: any }) {
  const groups = groupedSources(response);
  const order: SourceGroup[] = ['internal', 'external', 'web'];
  if (!order.some((k) => groups[k].length)) return null;
  return (
    <div className="srcs">
      <h4>Sources used</h4>
      {order.filter((k) => groups[k].length).map((k) => (
        // Collapsed by default: three groups expanded pushed the answer far up the panel, and
        // sources are a thing you consult, not a thing you read. <details> matches the Reasoning
        // block above and gives keyboard + screen-reader behaviour for free.
        <details key={k} className="grp">
          <summary className="hd">{GROUP_LABEL[k]}<span className="n">{groups[k].length} item{groups[k].length === 1 ? '' : 's'}</span><span className="chev">▾</span></summary>
          {groups[k].slice(0, 12).map((s, i) => {
            const title = String(s.title || s.doc_id || '(untitled)');
            // sourceHref, not s.url: internal knowledge elements carry no url of their own and
            // would otherwise render as plain text while external hits beside them are links.
            const url = sourceHref(s);
            const snip = String(s.abstract || s.snippet || s.contents || '').trim();
            return (
              <div key={i} className="it">
                <div className="t">{/^https?:\/\//i.test(url) ? <a href={url} target="_blank" rel="noopener noreferrer">{title}</a> : title}</div>
                {snip && <div className="sn">{snip.length > 260 ? snip.slice(0, 260) + '…' : snip}</div>}
              </div>
            );
          })}
          {groups[k].length > 12 && <div className="sn">+{groups[k].length - 12} more not shown</div>}
        </details>
      ))}
    </div>
  );
}

function AgentTurn({ m, resolveUrl }: { m: ChatMessage; resolveUrl: (u: string) => string }) {
  const imgs = (m.artifacts || []).filter(isImg).filter((f) => !(m.html || '').includes(f.file_id));
  const files = (m.artifacts || []).filter((f) => !isImg(f));
  const hasBody = m.html || m.text;
  return (
    <div className="turn">
      <div className="ai-label">I-GUIDE AI{m.streaming && <span className="spin" />}</div>
      {m.trace && m.trace.length > 0 && (
        <details className="reason" open={m.streaming}>
          <summary>Reasoning<span className="tally">{m.streaming ? 'thinking…' : `${m.trace.length} steps`}</span><span className="chev">▾</span></summary>
          <div className="body">{m.trace.map((t, j) => <div key={j} className={`ln ${t.kind || ''}`}>{t.text}</div>)}</div>
        </details>
      )}
      {(hasBody || imgs.length > 0 || m.response) && (
        <div className="answer-card">
          {m.html ? <div className="md" dangerouslySetInnerHTML={{ __html: m.html }} /> : m.text ? <div className="md"><p>{m.text}</p></div> : null}
          {m.streaming && !m.html && <span className="cursor">▋</span>}
          {imgs.length > 0 && (
            <div className="art-imgs">
              {imgs.map((f) => (
                <figure className="art" key={f.file_id}>
                  <a href={resolveUrl(f.download_url)} target="_blank" rel="noopener noreferrer"><img src={resolveUrl(f.download_url)} alt={f.filename} loading="lazy" /></a>
                  <figcaption><span className="nm">{f.filename}</span></figcaption>
                </figure>
              ))}
            </div>
          )}
          {files.length > 0 && <div className="files">{files.map((f) => <a key={f.file_id} href={resolveUrl(f.download_url)} target="_blank" rel="noopener noreferrer">{f.filename}</a>)}</div>}
          {m.response && <Sources response={m.response} />}
        </div>
      )}
    </div>
  );
}

// How far from the bottom still counts as being AT the bottom. Never assume exactly 0:
// sub-pixel rounding and fractional device pixel ratios leave a residue of a pixel or two, and
// a reader a hair off the bottom still means "keep following".
const BOTTOM_SLACK_PX = 32;

export function ChatPanel(p: Props) {
  const [text, setText] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  // Whether the transcript is FOLLOWING new content. True while the reader is at the bottom,
  // false once they scroll up to read something. A ref, not state: it changes on every scroll
  // event and nothing renders from it, so re-rendering the transcript on each one would be
  // pure waste during a stream.
  const pinnedRef = useRef(true);
  // Until when a scroll may be treated as the READER's. Not every scroll event is one: the
  // browser re-anchors the scroll position by itself as streaming content reflows above the
  // viewport, and an earlier version of this took those for the reader scrolling away and
  // detached — permanently, because only a scroll back to the bottom re-attaches, and the
  // reader had never scrolled at all. So a scroll can only DETACH while the reader is
  // demonstrably driving; anything else can only ever re-attach.
  const drivingUntilRef = useRef(0);

  // Generous, to cover trackpad momentum after the last wheel event. Harmless if it is too
  // long: the only scrolls this window admits are ones that land away from the bottom, and a
  // programmatic follow always lands AT it.
  const markDriving = () => { drivingUntilRef.current = Date.now() + 1200; };

  const isAtBottom = () => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_SLACK_PX;
  };

  const onScroll = () => {
    // Landing at the bottom always re-attaches, whoever caused it.
    if (isAtBottom()) { pinnedRef.current = true; return; }
    // Away from the bottom detaches only if the reader put it there.
    if (Date.now() < drivingUntilRef.current) pinnedRef.current = false;
  };

  useEffect(() => {
    const el = scrollRef.current;
    // Follow only when the reader is already at the bottom. This used to scroll
    // unconditionally, and with the full trace on it fires constantly — every reasoning step
    // patches `messages` with a new array — so reading anything above the fold was impossible:
    // the view was dragged back down mid-sentence every few hundred milliseconds. It fired
    // even with the reasoning block COLLAPSED, because the steps still change the array
    // whether or not anything visible grew.
    if (!el || !pinnedRef.current) return;
    // Instant, not smooth. A smooth scroll animates THROUGH positions that are not at the
    // bottom, and the handler below would read those as the reader moving away and unpin — so
    // the next step would silently stop following. Nothing looks smooth anyway when steps
    // arrive faster than the animation can finish.
    el.scrollTop = el.scrollHeight;
  }, [p.messages, p.busy]);

  const send = (t: string) => {
    const v = t.trim();
    if (!v || p.busy) return;
    setText('');
    // Sending re-attaches. You asked the question; you want to watch the answer arrive, even
    // if you had scrolled up to re-read something before hitting enter.
    pinnedRef.current = true;
    p.onSend(v);
  };

  return (
    <section className="chat">
      {p.showSettings && (
        <div className="settings">
          <div className="grid">
            <label>Mode
              <select value={p.mode} onChange={(e) => p.onSetMode(e.target.value as Mode)}>
                <option value="live">Live agent (real backend)</option>
                <option value="local">Local demo (mock, offline)</option>
              </select>
            </label>
            <label>API key
              <input type="password" value={p.cfg.apiKey} placeholder="X-API-KEY (if required)" onChange={(e) => p.onSetCfg({ ...p.cfg, apiKey: e.target.value })} />
            </label>
            <label>Model
              <select value={p.cfg.model || ''}
                      onChange={(e) => {
                        const model = e.target.value;
                        // Carry the provider alongside the id: two providers could serve
                        // similarly-named models, and the server should not have to guess.
                        const owner = p.models?.providers.find(
                          (g: ModelCatalogue['providers'][number]) => g.models.includes(model));
                        // Repair the effort on switch. Leaving a stale value behind is how a
                        // pick of 'high' on one model turned every later turn into a 400 on a
                        // model that refuses any level once tools are attached — and the value
                        // persists to localStorage, so it outlived the reload too.
                        const legal = owner?.effort_options?.[model] || [];
                        const forced = owner?.effort_required?.[model];
                        const kept = forced
                          ? forced
                          : (p.cfg.reasoningEffort && legal.includes(p.cfg.reasoningEffort)
                              ? p.cfg.reasoningEffort : '');
                        p.onSetCfg({ ...p.cfg, model, provider: model ? (owner?.provider || '') : '',
                                     reasoningEffort: kept });
                      }}>
                <option value="">
                  Agent default{p.models ? ` (${p.models.default.model})` : ''}
                </option>
                {/* An unconfigured provider is shown DISABLED rather than dropped. Absent
                    reads as "this deployment cannot speak Claude", which is a different
                    thing from "nobody has put a key in yet" — and only one of those is
                    fixable from the .env. */}
                {(p.models?.providers || []).map((g: ModelCatalogue['providers'][number]) => (
                  <optgroup key={g.provider} disabled={!g.configured}
                    label={g.label
                      + (g.configured ? (g.caveat ? ` — ${g.caveat}` : '')
                                      : ` — needs ${g.needs || 'configuration'}`)
                      + (g.stale ? ' — list unavailable' : '')}>
                    {g.models.map((m: string) => (
                      <option key={m} value={m} disabled={!g.configured}>{m}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>
            {/* The legal efforts depend on the model AND on tools being attached, which they
                always are here. Offer exactly what the API accepts: a model with one forced
                value is shown as fixed, and a model with no options shows no control. */}
            {(() => {
              const owner = p.models?.providers.find(
                (g: ModelCatalogue['providers'][number]) => g.models.includes(p.cfg.model || ''));
              const legal = owner?.effort_options?.[p.cfg.model || ''] || [];
              const forced = owner?.effort_required?.[p.cfg.model || ''];
              if (!legal.length) return null;
              if (forced) {
                return (
                  <label>Reasoning effort
                    <select value={forced} disabled title={`${p.cfg.model} requires reasoning_effort='${forced}' when tools are attached`}>
                      <option value={forced}>{forced} (required)</option>
                    </select>
                  </label>
                );
              }
              return (
                <label>Reasoning effort
                  <select value={p.cfg.reasoningEffort || ''}
                          onChange={(e) => p.onSetCfg({ ...p.cfg, reasoningEffort: e.target.value })}>
                    <option value="">Model default</option>
                    {legal.map((v: string) => <option key={v} value={v}>{v}</option>)}
                  </select>
                </label>
              );
            })()}
            {/* Which SHAPE runs the turn: the supervisor routing between a search peer and
                an analyze peer, or one agent doing both in one context. Per-request so the
                two can be compared side by side rather than by restarting the deployment. */}
            <label>Orchestration
              <select value={p.cfg.orchestration || ''}
                      onChange={(e) => p.onSetCfg({ ...p.cfg, orchestration: e.target.value })}>
                <option value="">Server default</option>
                <option value="peers">Supervisor + peers</option>
                <option value="unified">Unified agent (experimental)</option>
              </select>
            </label>
            {/* A SECOND axis, deliberately its own control: `Model` picks what writes
                the answer, this picks what writes the code. A peer whose sandbox image or
                credential is missing is shown but disabled — absent from the list would
                read as "not built", which is a different problem from "not configured". */}
            {p.models?.code_peers && (
              <label>Code peer
                <select value={p.cfg.codePeer || ''}
                        onChange={(e) => {
                          // Drop a model chosen for the previous peer. Carrying 'opus'
                          // onto a backend that never heard of it is how a stale
                          // reasoning_effort used to 400 every later turn.
                          p.onSetCfg({ ...p.cfg, codePeer: e.target.value, codePeerModel: '' });
                        }}>
                  <option value="">
                    Server default ({p.models.code_peers.default})
                  </option>
                  {p.models.code_peers.peers.map((peer) => (
                    <option key={peer.id} value={peer.id} disabled={!peer.available}
                            title={peer.label}>
                      {/* Just the peer. Its model is the control next to this one, and
                          showing "claude (sonnet)" here read as a fixed pairing. */}
                      {peer.id}
                      {peer.available ? '' : ` — ${peer.reason || 'unavailable'}`}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {/* Only for a peer that HAS selectable models, and only once one is chosen:
                the built-in peer codes with whatever `Model` above already picked. */}
            {(() => {
              const peer = p.models?.code_peers?.peers.find((x) => x.id === p.cfg.codePeer);
              if (!peer?.models?.length) return null;
              return (
                <label>Peer model
                  <select value={p.cfg.codePeerModel || ''}
                          onChange={(e) => p.onSetCfg({ ...p.cfg, codePeerModel: e.target.value })}>
                    <option value="">Peer default{peer.model ? ` (${peer.model})` : ''}</option>
                    {peer.models.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </label>
              );
            })()}
            <label className="wide">Chat endpoint
              <input value={p.cfg.endpoint} onChange={(e) => p.onSetCfg({ ...p.cfg, endpoint: e.target.value })} />
            </label>
          </div>
          <label className="chk"><input type="checkbox" checked={p.spatial} onChange={(e) => p.onSetSpatial(e.target.checked)} /> Spatial tools (maps, OSM/Overpass, geo search) — off = pure chat</label>
        </div>
      )}

      {p.spatial && (
        <>
          <div className="toolbar">
            {p.mapVisible && <span className="hint">right-drag the map to select a region</span>}
            {/* Shown only when there IS a region: a permanently greyed-out button is just
                clutter — the toolbar should offer what can actually be done right now. */}
            {p.hasRegion && <button onClick={p.onClearRegion}>Clear region</button>}
            <span className={p.hasRegion ? 'rstat on' : 'rstat'}>{p.hasRegion ? '● region set' : '◇ spatial on'}</span>
          </div>
        </>
      )}

      <div className="transcript" ref={scrollRef}
        onScroll={onScroll}
        onWheel={markDriving} onTouchMove={markDriving}
        onMouseDown={markDriving} onKeyDown={markDriving}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); const fs = Array.from(e.dataTransfer.files || []); if (fs.length) p.onUpload(fs); }}>
        {p.messages.map((m, i) => m.role === 'user' ? (
          <div className="turn user" key={i}>
            <div className="who you">You</div>
            <div className="row right"><div className="bubble user">{m.text}</div></div>
          </div>
        ) : <AgentTurn key={i} m={m} resolveUrl={p.resolveUrl} />)}
        {p.busy && !p.messages.some((m) => m.streaming) && <div className="turn"><div className="ai-label">I-GUIDE AI<span className="spin" /></div></div>}
      </div>

      {p.messages.length <= 1 && p.tab !== 'rs' && (
        <div className="suggest">{SUGGESTIONS.map((s) => <button key={s} className="chip" onClick={() => send(s)}>{s}</button>)}</div>
      )}

      {/* The satellite-embedding operations, directly above the composer — where the eye
          already is when you go to type. Above the transcript they scrolled off the top of a
          long conversation and were never seen again.

          On the REMOTE SENSING tab they are always on screen, disabled until a region exists,
          with the two steps spelled out: that tab exists to SHOW what can be done, and a
          hidden control demonstrates nothing. Elsewhere they still appear only once a region
          is drawn — a permanently greyed row is clutter in a tab that is not about them. */}
      {p.spatial && (p.tab === 'rs' || p.hasRegion) && (
        <div className={`rspanel ${p.tab === 'rs' ? 'demo' : ''}`}>
          {p.tab === 'rs' && (
            <ol className="rssteps">
              <li className={p.hasRegion ? 'done' : ''}>
                {p.mapVisible ? 'Right-drag on the map to draw a region'
                              : 'Open the map, then right-drag to draw a region'}
              </li>
              <li className={p.hasRegion ? '' : 'muted'}>Pick an operation</li>
            </ol>
          )}
          <div className="rsrow">
            {/* The numbered steps above already say what this row is, and at a 460px panel the
                label costs 126px — exactly enough to push the fourth operation onto a second
                line. Kept where there are no steps to explain it. */}
            {p.tab !== 'rs' && <span className="rslabel">🛰 satellite embedding</span>}
            {RS_ACTIONS.map((a) => (
              <button key={a.label} className="rsbtn" disabled={p.busy || !p.hasRegion}
                title={p.hasRegion ? a.prompt : 'Draw a region on the map first'}
                onClick={() => p.onSend(a.prompt)}>{a.label}</button>
            ))}
          </div>
        </div>
      )}

      <div className="composer">
        <div className="box">
          {/* Map lives with the composer controls, left of Attach: it is a thing you reach
              for while composing, and selecting a region needs the map open first. */}
          <button type="button" className={`circle map ${p.mapVisible ? 'on' : ''}`}
                  onClick={p.onToggleMap} aria-label={p.mapVisible ? 'Hide map' : 'Show map'}
                  title={p.mapVisible ? 'Hide the map' : 'Show the map, then right-drag on it to select a region'}>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 3.5 3.4 5.6a1 1 0 0 0-.65.94v13.2a.7.7 0 0 0 .95.65L9 18.5l6 2 5.6-2.1a1 1 0 0 0 .65-.94V4.26a.7.7 0 0 0-.95-.65L15 5.5Z" />
              <path d="M9 3.5v15M15 5.5v15" />
            </svg>
          </button>
          <label className="circle attach" title="Attach files">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.5l-8.5 8.5a5 5 0 01-7-7l9-9a3.5 3.5 0 015 5l-9 9a2 2 0 01-3-3l8-8" /></svg>
            <input type="file" multiple style={{ display: 'none' }} onChange={(e) => { const fs = Array.from(e.target.files || []); if (fs.length) p.onUpload(fs); (e.target as HTMLInputElement).value = ''; }} />
          </label>
          <textarea value={text}
            placeholder={p.mode === 'live' ? 'Ask me anything…' : 'Offline demo — try “show hospitals here”'}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(text); } }} />
          {p.busy ? (
            <button className="circle stop" onClick={p.onStop} title="Stop the agent" aria-label="Stop">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
            </button>
          ) : (
            <button className="circle send" onClick={() => send(text)} disabled={!text.trim()} title="Send" aria-label="Send">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 11l5-5 5 5M12 6v12" /></svg>
            </button>
          )}
        </div>
        <div className="footline">
          <button className="conn" onClick={p.onToggleSettings}>⚙ Connection</button>
          <span className="terms">I-GUIDE Platform Terms of Use apply. Smart Search can make mistakes. Always double-check.</span>
        </div>
      </div>
    </section>
  );
}
