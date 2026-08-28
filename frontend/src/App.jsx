import { useEffect, useState } from "react";
import { connectPlateStream } from "./lib/ws.js";
import PlateView from "./components/PlateView.jsx";
import WellDetail from "./components/WellDetail.jsx";
import AlertFeed from "./components/AlertFeed.jsx";
import RecommendationPanel from "./components/RecommendationPanel.jsx";
import ExplainerPanel from "./components/ExplainerPanel.jsx";
import CalibrationPanel from "./components/CalibrationPanel.jsx";
import PipelineMap from "./components/PipelineMap.jsx";
import PageIntro from "./components/PageIntro.jsx";
import { DishIcon, NetworkIcon, SlidersIcon, FlaskIcon, DatabaseIcon } from "./components/Icons.jsx";
import ControlArenaTab from "./tabs/ControlArenaTab.jsx";
import ControlPanelTab from "./tabs/ControlPanelTab.jsx";
import DrugScreeningTab from "./tabs/DrugScreeningTab.jsx";
import DataUploadTab from "./tabs/DataUploadTab.jsx";
import LandingPage from "./pages/LandingPage.jsx";

const HISTORY_LEN = 180; // ticks kept per well (~90 simulated hours at 30min/step)
const EVENT_LOG_LEN = 50;
const CALIBRATION_LEN = 150;

const TABS = [
  { id: "monitor", label: "Live Monitor", icon: DishIcon },
  { id: "arena", label: "Control Arena", icon: NetworkIcon },
  { id: "panel", label: "Control Panel", icon: SlidersIcon },
  { id: "drug", label: "Drug Screening", icon: FlaskIcon },
  { id: "upload", label: "Data Upload", icon: DatabaseIcon },
];

function pushHistory(prev, plateState) {
  const next = { ...prev };
  for (const w of plateState.wells) {
    const h = next[w.well_id]
      ? { ...next[w.well_id] }
      : { t: [], ph: [], do2: [], glucose_lactate: [], impedance: [], health_score: [], health_std: [] };
    h.t = [...h.t, w.timestamp].slice(-HISTORY_LEN);
    h.ph = [...h.ph, w.reading.ph].slice(-HISTORY_LEN);
    h.do2 = [...h.do2, w.reading.do2].slice(-HISTORY_LEN);
    h.glucose_lactate = [...h.glucose_lactate, w.reading.glucose_lactate].slice(-HISTORY_LEN);
    h.impedance = [...h.impedance, w.reading.impedance].slice(-HISTORY_LEN);
    h.health_score = [...h.health_score, w.health_score].slice(-HISTORY_LEN);
    h.health_std = [...h.health_std, w.health_std].slice(-HISTORY_LEN);
    next[w.well_id] = h;
  }
  return next;
}

export default function App() {
  const [view, setView] = useState("landing"); // "landing" | "dashboard"
  const [tab, setTab] = useState("monitor");
  const [plateState, setPlateState] = useState(null);
  const [history, setHistory] = useState({});
  const [events, setEvents] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [calibrationHistory, setCalibrationHistory] = useState([]);
  const [status, setStatus] = useState("connecting");
  const [selectedWell, setSelectedWell] = useState(null);

  useEffect(() => {
    const close = connectPlateStream({
      onMessage: (frame) => {
        setPlateState(frame.plate_state);
        setHistory((prev) => pushHistory(prev, frame.plate_state));
        if (frame.new_events?.length) {
          setEvents((prev) => [...frame.new_events, ...prev].slice(0, EVENT_LOG_LEN));
        }
        setRecommendations(frame.recommendations ?? []);
        if (frame.calibration) {
          setCalibrationHistory((prev) => [...prev, frame.calibration].slice(-CALIBRATION_LEN));
        }
      },
      onStatusChange: setStatus,
    });
    return close;
  }, []);

  const selected = plateState?.wells.find((w) => w.well_id === selectedWell) ?? null;
  const selectedHistory = selectedWell ? history[selectedWell] : null;

  if (view === "landing") {
    return (
      <div>
        <LandingPage onEnter={() => setView("dashboard")} />
      </div>
    );
  }

  return (
    <div>
      <header className="app-header">
        <span
          className="app-mark"
          onClick={() => setView("landing")}
          title="Back to overview"
          style={{ cursor: "pointer" }}
        >
          <DishIcon style={{ color: "#fff", width: 22, height: 22 }} />
        </span>
        <div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.6rem" }}>
            <h1
              className="app-title"
              onClick={() => setView("landing")}
              title="Back to overview"
              style={{ cursor: "pointer" }}
            >
              OrganoidTwin
            </h1>
            <span className={`status-pill ${status}`}>{status}</span>
          </div>
          <p style={{ margin: "0.1rem 0 0", fontSize: "0.82rem", color: "var(--text-muted)" }}>
            AI-driven multimodal monitoring &amp; adaptive control for organoid culture plates
          </p>
        </div>
      </header>

      <nav className="tab-nav">
        {TABS.map((t) => {
          const Icon = t.icon;
          return (
            <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)} style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <Icon style={{ width: 15, height: 15 }} />
              {t.label}
            </button>
          );
        })}
      </nav>

      <PipelineMap />

      {tab === "monitor" && (
        <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 1.5rem 1.5rem" }}>
          <PageIntro
            icon={DishIcon}
            title="Live Monitor"
            tagline="Watch a simulated organoid culture, live, as the model sees it."
            description={
              <>
                This replays a simulated 7-day organoid culture, one 30-minute timestep at a time. Each
                circle on the plate is a well; its color is the model's inferred health, not the ground
                truth (you're seeing what a real deployment would see: noisy sensor readings in, a
                health score out). A dashed ring means the model is uncertain about that well. When the
                bifurcation detector fires, it shows up in the Alert Feed and a Recommendation appears.
              </>
            }
            pipeline={[
              { label: "Metabolic sim", info: "Generates the hidden ground truth for every well every tick." },
              { label: "Sensors", info: "Turns that ground truth into the noisy pH / O2 / glucose-lactate / impedance readings actually shown." },
              { label: "GATv2 GNN", info: "Fuses the 4 sensor streams across the whole plate into a health score + uncertainty, every tick." },
              { label: "Bifurcation detector", info: "Runs every tick; fires an Alert Feed entry when a well's prediction sensitivity spikes." },
              { label: "Recommendation + narrator", info: "Turns a flagged well's sensor trend into a concrete action and a plain-language sentence." },
            ]}
            tryItems={[
              "Click any well to see its 4 raw sensor traces, health score, and plain-language status.",
              "Watch the Alert Feed — a regime-shift firing means the bifurcation detector just caught something.",
              "Check the Calibration panel to see the model quietly adapting via periodic EWC fine-tuning.",
            ]}
          />

          <div className="app-shell" style={{ padding: "1.5rem 0 0" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              <PlateView
                plateState={plateState}
                selectedWell={selectedWell}
                onSelectWell={setSelectedWell}
              />
              <ExplainerPanel well={selected} />
              <AlertFeed events={events} onSelectWell={setSelectedWell} />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              <WellDetail well={selected} history={selectedHistory} />
              <RecommendationPanel recommendations={recommendations} selectedWell={selectedWell} onSelectWell={setSelectedWell} />
              <CalibrationPanel history={calibrationHistory} />
            </div>
          </div>
        </div>
      )}

      {tab === "arena" && <ControlArenaTab />}
      {tab === "panel" && <ControlPanelTab plateState={plateState} />}
      {tab === "drug" && <DrugScreeningTab />}
      {tab === "upload" && <DataUploadTab />}
    </div>
  );
}
