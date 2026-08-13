import React, { useState, useEffect, useCallback } from "react";
import {
  Search,
  Sparkles,
  SlidersHorizontal,
  X,
  Clock,
  Globe,
  Gauge,
  Newspaper,
  GraduationCap,
  Code2,
  CornerDownLeft,
} from "lucide-react";

/* =========================================================================
   MOCK DATA LAYER
   Swap `mockSearch` for a real call to your Python backend. Keep the
   returned shape the same and nothing else in this file needs to change:

   async function mockSearch(query, filter) {
     const res = await fetch("/api/search", {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ query, filter }),
     });
     return res.json(); // -> [{ id, title, summary, sourceType, domain, confidence, timestamp }]
   }
   ========================================================================= */

const FILTERS = [
  { id: "all", label: "All sources", icon: Sparkles },
  { id: "web", label: "Web", icon: Globe },
  { id: "news", label: "News", icon: Newspaper },
  { id: "academic", label: "Academic", icon: GraduationCap },
  { id: "code", label: "Code", icon: Code2 },
];
const FILTER_LABEL = Object.fromEntries(FILTERS.map((f) => [f.id, f.label]));

const SOURCE_META = {
  web: { label: "Web", color: "#4FE3FF" },
  news: { label: "News", color: "#9B6BFF" },
  academic: { label: "Academic", color: "#5C8DFF" },
  code: { label: "Code", color: "#5CFFB3" },
};

const DOMAINS = {
  web: ["nova.io", "gridwire.net", "openlens.dev"],
  news: ["dailysignal.news", "the-current.press", "wire.report"],
  academic: ["arxiv.org", "nature-review.org", "openscience.edu"],
  code: ["github.com", "devnotes.io", "changelog.dev"],
};

const SUGGESTIONS = [
  "Explain quantum entanglement simply",
  "Latest breakthroughs in fusion energy",
  "React vs Svelte in 2026",
  "How do mRNA vaccines work",
];

const TITLE_TEMPLATES = [
  "What we know about",
  "Inside",
  "A closer look at",
  "Understanding",
  "The state of",
];

const SUMMARY_SENTENCES = [
  "Recent findings have shifted the consensus view, prompting a re-examination of older assumptions.",
  "The core idea rests on a handful of well-tested principles, though details vary a lot by context.",
  "Several independent teams have converged on similar conclusions, which strengthens overall confidence.",
  "Open questions remain, particularly around long-term effects and how results generalize outside the lab.",
  "Practical applications are already emerging, with early adopters reporting gains within months.",
];

function pick(arr, seed) {
  return arr[((seed % arr.length) + arr.length) % arr.length];
}

function buildSummary(len) {
  return SUMMARY_SENTENCES.slice(0, len).join(" ");
}

function mockSearch(query, filter) {
  return new Promise((resolve) => {
    setTimeout(() => {
      if (query.toLowerCase().includes("empty")) {
        resolve([]);
        return;
      }
      const types = filter === "all" ? ["web", "news", "academic", "code"] : [filter, "web", "academic"];
      const results = Array.from({ length: 5 }).map((_, i) => {
        const type = pick(types, i);
        const domain = pick(DOMAINS[type], i + 2);
        const len = [2, 4, 3, 5, 3][i % 5];
        return {
          id: `${Date.now()}-${i}`,
          title: `${pick(TITLE_TEMPLATES, i)} ${query}`,
          summary: buildSummary(len),
          sourceType: type,
          domain,
          confidence: 62 + ((i * 17) % 36),
          timestamp: ["3m ago", "1h ago", "6h ago", "yesterday", "2d ago"][i % 5],
        };
      });
      resolve(results);
    }, 1500);
  });
}

/* =========================================================================
   SUBCOMPONENTS
   ========================================================================= */

function TypingText({ text, reducedMotion, speed = 14 }) {
  const [shown, setShown] = useState(reducedMotion ? text : "");

  useEffect(() => {
    if (reducedMotion) {
      setShown(text);
      return;
    }
    setShown("");
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      setShown(text.slice(0, i));
      if (i >= text.length) clearInterval(id);
    }, speed);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, reducedMotion]);

  const done = shown.length >= text.length;

  return (
    <p className="lucid-summary">
      {shown}
      {!done && <span className="lucid-caret" aria-hidden="true" />}
    </p>
  );
}

function ResultCard({ result, index, reducedMotion }) {
  const meta = SOURCE_META[result.sourceType];
  const confColor =
    result.confidence >= 90 ? "#4FE3FF" : result.confidence >= 75 ? "#9B6BFF" : "#B8BCC4";

  return (
    <div
      className="lucid-card glass"
      style={{ animationDelay: reducedMotion ? "0ms" : `${index * 90}ms` }}
    >
      <div className="lucid-card-head">
        <span className="lucid-source-dot" style={{ background: meta.color }} />
        <span>{meta.label}</span>
        <span className="lucid-dot-sep">•</span>
        <span className="lucid-timestamp">
          <Clock size={12} style={{ marginRight: 4 }} />
          {result.timestamp}
        </span>
      </div>
      <h3 className="lucid-card-title">{result.title}</h3>
      <TypingText text={result.summary} reducedMotion={reducedMotion} />
      <div className="lucid-card-foot">
        <span className="lucid-confidence" style={{ color: confColor, borderColor: `${confColor}55` }}>
          <Gauge size={12} />
          {result.confidence}% confidence
        </span>
        <span className="lucid-domain">{result.domain}</span>
      </div>
    </div>
  );
}

function SkeletonCard({ index, reducedMotion }) {
  const widths = ["92%", "78%", "85%", "60%"];
  return (
    <div
      className="lucid-skel glass"
      style={{ animationDelay: reducedMotion ? "0ms" : `${index * 90}ms` }}
    >
      <div className="lucid-skel-line" style={{ width: "40%", height: 10 }} />
      <div className="lucid-skel-line" style={{ width: "70%", height: 16, marginTop: 14 }} />
      {widths.map((w, i) => (
        <div key={i} className="lucid-skel-line" style={{ width: w }} />
      ))}
    </div>
  );
}

/* =========================================================================
   MAIN COMPONENT
   ========================================================================= */

export default function LucidSearch() {
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState(null); // null = not searched yet, [] = no matches, [...] = matches
  const [filter, setFilter] = useState("all");
  const [sheetOpen, setSheetOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const handler = (e) => setReducedMotion(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const runSearch = useCallback(
    (value) => {
      const q = (value ?? query).trim();
      if (!q) return;
      setQuery(q);
      setHasSearched(true);
      setIsSearching(true);
      setResults(null);
      setSheetOpen(false);
      mockSearch(q, filter).then((data) => {
        setResults(data);
        setIsSearching(false);
      });
    },
    [query, filter]
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    runSearch();
  };

  const statusText = isSearching
    ? "Searching…"
    : results
    ? `${results.length} result${results.length === 1 ? "" : "s"} found`
    : "";

  return (
    <div className="lucid-root">
      <style>{CSS}</style>

      <div className="lucid-bg" aria-hidden="true">
        <div className="lucid-blob lucid-blob-a" />
        <div className="lucid-blob lucid-blob-b" />
      </div>

      <header className="lucid-header">
        <div className="lucid-brand">
          <Sparkles size={18} />
          <span>LUCID</span>
        </div>
        <div className="lucid-status">
          <span className="lucid-status-dot" />
          Online
        </div>
      </header>

      <main className={`lucid-main ${hasSearched ? "lucid-main--docked" : "lucid-main--hero"}`}>
        {!hasSearched && (
          <div className="lucid-hero-copy">
            <h1 className="lucid-title">
              Ask anything.
              <br />
              <span className="lucid-title-shine">Get clarity.</span>
            </h1>
            <p className="lucid-subtitle">
              Lucid reads across the web, papers, and code to answer in plain language —
              with sources you can check.
            </p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="lucid-searchwrap">
          <div
            className={`lucid-orb ${
              isSearching ? "lucid-orb--searching" : isFocused ? "lucid-orb--focus" : "lucid-orb--idle"
            }`}
            aria-hidden="true"
          />
          <div
            className={`lucid-searchbar glass ${isFocused ? "lucid-searchbar--focus" : ""} ${
              isSearching ? "lucid-searchbar--searching" : ""
            }`}
          >
            <Search size={18} className="lucid-search-icon" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Ask Lucid anything…"
              className="lucid-input"
              aria-label="Search query"
            />
            <button
              type="button"
              className="lucid-filter-btn"
              onClick={() => setSheetOpen(true)}
              aria-label="Open source filters"
            >
              <SlidersHorizontal size={16} />
            </button>
            <button
              type="submit"
              className="lucid-submit-btn"
              disabled={!query.trim() || isSearching}
              aria-label="Run search"
            >
              {isSearching ? <span className="lucid-spinner" /> : <CornerDownLeft size={16} />}
            </button>
          </div>
        </form>

        <span className="lucid-sr-only" role="status" aria-live="polite">
          {statusText}
        </span>

        {!hasSearched && (
          <div className="lucid-suggestions">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="lucid-chip" onClick={() => runSearch(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        {hasSearched && (
          <div className="lucid-results-area">
            {isSearching && (
              <div className="lucid-masonry">
                {[0, 1, 2, 3].map((i) => (
                  <SkeletonCard key={i} index={i} reducedMotion={reducedMotion} />
                ))}
              </div>
            )}

            {!isSearching && results && results.length > 0 && (
              <div className="lucid-masonry">
                {results.map((r, i) => (
                  <ResultCard key={r.id} result={r} index={i} reducedMotion={reducedMotion} />
                ))}
              </div>
            )}

            {!isSearching && results && results.length === 0 && (
              <div className="lucid-empty glass">
                <Search size={22} />
                <h3>No signal on that one.</h3>
                <p>
                  Try different words, or widen your filters — right now you're only searching{" "}
                  {FILTER_LABEL[filter]}.
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {sheetOpen && (
        <div className="lucid-sheet-overlay" onClick={() => setSheetOpen(false)}>
          <div className="lucid-sheet glass" onClick={(e) => e.stopPropagation()}>
            <div className="lucid-sheet-handle" />
            <div className="lucid-sheet-head">
              <span>Search sources</span>
              <button onClick={() => setSheetOpen(false)} aria-label="Close filters">
                <X size={18} />
              </button>
            </div>
            <div className="lucid-sheet-options">
              {FILTERS.map((f) => {
                const Icon = f.icon;
                return (
                  <button
                    key={f.id}
                    className={`lucid-sheet-option ${filter === f.id ? "is-active" : ""}`}
                    onClick={() => setFilter(f.id)}
                  >
                    <Icon size={16} />
                    {f.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* =========================================================================
   STYLES
   Tokens — bg #050505 · blue #2F6BFF · violet #8A4FFF · cyan #4FE3FF
   Display: Space Grotesk · Body: Inter · Data/mono: JetBrains Mono
   ========================================================================= */

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

.lucid-root{position:relative;min-height:100vh;background:#050505;color:#F4F5F7;font-family:'Inter',sans-serif;overflow-x:hidden;-webkit-font-smoothing:antialiased;}
.lucid-sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;}

.lucid-bg{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;}
.lucid-blob{position:absolute;border-radius:50%;filter:blur(90px);}
.lucid-blob-a{width:60vw;height:60vw;top:-15%;left:-15%;background:radial-gradient(circle, rgba(47,107,255,.35), transparent 70%);animation:lucid-drift-a 26s ease-in-out infinite;}
.lucid-blob-b{width:55vw;height:55vw;bottom:-20%;right:-15%;background:radial-gradient(circle, rgba(138,79,255,.30), transparent 70%);animation:lucid-drift-b 32s ease-in-out infinite;}

.glass{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.10);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);box-shadow:inset 0 1px 0 rgba(255,255,255,.06), 0 20px 40px -20px rgba(0,0,0,.6);}

.lucid-header{position:relative;z-index:2;display:flex;align-items:center;justify-content:space-between;padding:20px clamp(16px,4vw,48px);}
.lucid-brand{display:flex;align-items:center;gap:8px;font-family:'Space Grotesk',sans-serif;font-weight:600;letter-spacing:.04em;font-size:15px;}
.lucid-brand svg{color:#4FE3FF;}
.lucid-status{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(244,245,247,.5);letter-spacing:.03em;text-transform:uppercase;}
.lucid-status-dot{width:6px;height:6px;border-radius:50%;background:#4FE3FF;box-shadow:0 0 8px 2px rgba(79,227,255,.7);animation:lucid-orb-idle 3s ease-in-out infinite;}

.lucid-main{position:relative;z-index:2;display:flex;flex-direction:column;align-items:center;padding:0 20px 80px;}
.lucid-main--hero{min-height:calc(100vh - 76px);justify-content:center;padding-top:40px;}
.lucid-main--docked{padding-top:28px;justify-content:flex-start;}

.lucid-hero-copy{text-align:center;margin-bottom:36px;animation:lucid-fade-up .8s ease both;}
.lucid-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:clamp(2.2rem,6vw,4rem);line-height:1.08;letter-spacing:-.02em;}
.lucid-title-shine{background:linear-gradient(100deg,#4FE3FF 10%,#8A4FFF 45%,#F4F5F7 55%,#4FE3FF 90%);background-size:250% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:lucid-shine 6s linear infinite;}
.lucid-subtitle{margin:18px auto 0;color:rgba(244,245,247,.56);font-size:clamp(.95rem,2vw,1.08rem);max-width:520px;}

.lucid-searchwrap{position:relative;width:100%;max-width:680px;animation:lucid-fade-up .8s .1s ease both;}
.lucid-orb{position:absolute;left:50%;top:50%;width:120%;height:340%;transform:translate(-50%,-50%);border-radius:50%;background:radial-gradient(circle, rgba(79,227,255,.28), rgba(138,79,255,.16) 45%, transparent 72%);filter:blur(38px);z-index:-1;pointer-events:none;}
.lucid-orb--idle{animation:lucid-orb-idle 4.5s ease-in-out infinite;}
.lucid-orb--focus{animation:lucid-orb-focus 2.6s ease-in-out infinite;}
.lucid-orb--searching{animation:lucid-orb-searching 1.6s linear infinite;}

.lucid-searchbar{position:relative;display:flex;align-items:center;gap:10px;padding:8px 8px 8px 22px;border-radius:999px;transition:box-shadow .35s ease, border-color .35s ease;}
.lucid-searchbar--focus{border-color:rgba(79,227,255,.4);box-shadow:0 0 0 1px rgba(79,227,255,.35), 0 0 30px 6px rgba(79,227,255,.18), inset 0 1px 0 rgba(255,255,255,.08);}
.lucid-searchbar--searching{animation:lucid-pulse-border 1.6s ease-in-out infinite;}
.lucid-search-icon{color:rgba(244,245,247,.45);flex-shrink:0;}
.lucid-input{flex:1;background:transparent;border:none;outline:none;color:#F4F5F7;font-size:16px;font-family:'Inter',sans-serif;min-width:0;}
.lucid-input::placeholder{color:rgba(244,245,247,.35);}
.lucid-filter-btn,.lucid-submit-btn{display:flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:50%;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.04);color:rgba(244,245,247,.75);flex-shrink:0;cursor:pointer;transition:transform .25s ease, background .25s ease, box-shadow .25s ease;}
.lucid-filter-btn:hover,.lucid-submit-btn:hover:not(:disabled){transform:scale(1.08);background:rgba(255,255,255,.08);box-shadow:0 0 16px rgba(79,227,255,.25);}
.lucid-submit-btn{background:linear-gradient(135deg,#2F6BFF,#8A4FFF);color:#fff;border:none;}
.lucid-submit-btn:disabled{opacity:.35;cursor:not-allowed;}
.lucid-spinner{width:14px;height:14px;border-radius:50%;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;animation:lucid-spin .7s linear infinite;}

.lucid-chip:focus-visible,.lucid-submit-btn:focus-visible,.lucid-filter-btn:focus-visible,.lucid-sheet-option:focus-visible{outline:2px solid #4FE3FF;outline-offset:2px;}

.lucid-suggestions{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;max-width:640px;margin-top:26px;animation:lucid-fade-up .8s .2s ease both;}
.lucid-chip{font-family:'JetBrains Mono',monospace;font-size:12.5px;padding:9px 16px;border-radius:999px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.10);color:rgba(244,245,247,.68);cursor:pointer;transition:transform .25s ease, border-color .25s ease, color .25s ease;}
.lucid-chip:hover{transform:scale(1.05);border-color:rgba(79,227,255,.4);color:#F4F5F7;}

.lucid-results-area{width:100%;max-width:1180px;margin-top:44px;}
.lucid-masonry{column-count:1;column-gap:20px;}
@media(min-width:700px){.lucid-masonry{column-count:2;}}
@media(min-width:1080px){.lucid-masonry{column-count:3;}}

.lucid-card{break-inside:avoid;margin-bottom:20px;padding:22px;border-radius:22px;animation:lucid-fade-up .6s ease both;transition:transform .3s ease, box-shadow .3s ease, border-color .3s ease;}
.lucid-card:hover{transform:scale(1.02);border-color:rgba(255,255,255,.2);box-shadow:0 0 0 1px rgba(79,227,255,.2), 0 20px 50px -20px rgba(47,107,255,.35), inset 0 1px 0 rgba(255,255,255,.08);}
.lucid-card-head{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(244,245,247,.5);text-transform:uppercase;letter-spacing:.04em;margin-bottom:12px;}
.lucid-source-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.lucid-dot-sep{opacity:.4;}
.lucid-timestamp{display:flex;align-items:center;margin-left:auto;}
.lucid-card-title{font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:600;margin-bottom:10px;line-height:1.35;}
.lucid-summary{font-size:14px;line-height:1.65;color:rgba(244,245,247,.72);min-height:1.6em;}
.lucid-caret{display:inline-block;width:2px;height:14px;background:#4FE3FF;margin-left:2px;vertical-align:middle;animation:lucid-caret-blink .9s step-end infinite;}
.lucid-card-foot{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:16px;padding-top:14px;border-top:1px solid rgba(255,255,255,.08);flex-wrap:wrap;}
.lucid-confidence{display:flex;align-items:center;gap:5px;font-family:'JetBrains Mono',monospace;font-size:11px;padding:5px 10px;border-radius:999px;border:1px solid;background:rgba(255,255,255,.03);}
.lucid-domain{font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(244,245,247,.4);}

.lucid-skel{break-inside:avoid;margin-bottom:20px;padding:22px;border-radius:22px;animation:lucid-fade-up .5s ease both;}
.lucid-skel-line{height:11px;border-radius:6px;margin-bottom:10px;background:linear-gradient(90deg, rgba(255,255,255,.05) 25%, rgba(255,255,255,.14) 37%, rgba(255,255,255,.05) 63%);background-size:400px 100%;animation:lucid-shimmer 1.6s linear infinite;}

.lucid-empty{max-width:420px;margin:0 auto;padding:40px 30px;border-radius:24px;text-align:center;display:flex;flex-direction:column;align-items:center;gap:10px;animation:lucid-fade-up .5s ease both;}
.lucid-empty svg{color:rgba(244,245,247,.4);}
.lucid-empty h3{font-family:'Space Grotesk',sans-serif;font-size:17px;}
.lucid-empty p{font-size:13.5px;color:rgba(244,245,247,.55);line-height:1.6;}

.lucid-sheet-overlay{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.5);backdrop-filter:blur(4px);display:flex;align-items:flex-end;justify-content:center;animation:lucid-fade-in .2s ease both;}
.lucid-sheet{width:100%;max-width:420px;border-radius:28px 28px 0 0;padding:14px 20px 28px;animation:lucid-sheet-in .35s cubic-bezier(.2,.8,.2,1) both;}
.lucid-sheet-handle{width:40px;height:4px;border-radius:2px;background:rgba(255,255,255,.2);margin:4px auto 16px;}
.lucid-sheet-head{display:flex;align-items:center;justify-content:space-between;font-family:'Space Grotesk',sans-serif;font-weight:600;margin-bottom:14px;}
.lucid-sheet-head button{color:rgba(244,245,247,.6);background:none;border:none;cursor:pointer;}
.lucid-sheet-options{display:flex;flex-direction:column;gap:8px;}
.lucid-sheet-option{display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:14px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);color:rgba(244,245,247,.75);font-size:14px;text-align:left;cursor:pointer;transition:background .2s ease,border-color .2s ease;}
.lucid-sheet-option.is-active{background:linear-gradient(135deg, rgba(47,107,255,.2), rgba(138,79,255,.2));border-color:rgba(79,227,255,.4);color:#fff;}

@media(min-width:768px){
  .lucid-sheet-overlay{align-items:flex-start;justify-content:flex-end;padding:80px 24px 0 0;background:transparent;backdrop-filter:none;}
  .lucid-sheet{max-width:260px;border-radius:20px;animation:lucid-fade-up .25s ease both;}
}

@keyframes lucid-drift-a{0%{transform:translate(0,0) scale(1);}50%{transform:translate(6%,8%) scale(1.15);}100%{transform:translate(0,0) scale(1);}}
@keyframes lucid-drift-b{0%{transform:translate(0,0) scale(1);}50%{transform:translate(-8%,-6%) scale(1.1);}100%{transform:translate(0,0) scale(1);}}
@keyframes lucid-shine{0%{background-position:0% 0;}100%{background-position:250% 0;}}
@keyframes lucid-orb-idle{0%,100%{opacity:.55;transform:translate(-50%,-50%) scale(1);}50%{opacity:.85;transform:translate(-50%,-50%) scale(1.06);}}
@keyframes lucid-orb-focus{0%,100%{opacity:.9;transform:translate(-50%,-50%) scale(1.05);}50%{opacity:1;transform:translate(-50%,-50%) scale(1.16);}}
@keyframes lucid-orb-searching{0%{opacity:.9;transform:translate(-50%,-50%) scale(1) rotate(0deg);}50%{opacity:1;transform:translate(-50%,-50%) scale(1.25) rotate(180deg);}100%{opacity:.9;transform:translate(-50%,-50%) scale(1) rotate(360deg);}}
@keyframes lucid-pulse-border{0%,100%{box-shadow:0 0 0 1px rgba(79,227,255,.35), 0 0 24px 4px rgba(79,227,255,.18);}50%{box-shadow:0 0 0 1px rgba(138,79,255,.45), 0 0 34px 8px rgba(138,79,255,.28);}}
@keyframes lucid-fade-up{from{opacity:0;transform:translateY(14px) scale(.98);}to{opacity:1;transform:translateY(0) scale(1);}}
@keyframes lucid-fade-in{from{opacity:0;}to{opacity:1;}}
@keyframes lucid-shimmer{0%{background-position:-450px 0;}100%{background-position:450px 0;}}
@keyframes lucid-sheet-in{from{transform:translateY(100%);}to{transform:translateY(0);}}
@keyframes lucid-spin{to{transform:rotate(360deg);}}
@keyframes lucid-caret-blink{0%,49%{opacity:1;}50%,100%{opacity:0;}}

@media(prefers-reduced-motion: reduce){
  .lucid-blob,.lucid-orb,.lucid-title-shine,.lucid-status-dot,.lucid-searchbar--searching,.lucid-card,.lucid-hero-copy,.lucid-searchwrap,.lucid-suggestions,.lucid-skel-line,.lucid-caret,.lucid-empty,.lucid-sheet{animation:none !important;}
}
`;
