import { useState } from "react";
import PushupCounter from "./components/PushupCounter";
import History from "./components/History";
import "./App.css";

type Tab = "workout" | "history";

export default function App() {
  const [tab, setTab] = useState<Tab>("workout");

  return (
    <div className="app">
      <header className="header">
        <h1>💪 AI Pushup Counter</h1>
        <nav>
          <button className={tab === "workout" ? "active" : ""} onClick={() => setTab("workout")}>
            Workout
          </button>
          <button className={tab === "history" ? "active" : ""} onClick={() => setTab("history")}>
            History
          </button>
        </nav>
      </header>
      <main>
        {tab === "workout" ? <PushupCounter /> : <History />}
      </main>
    </div>
  );
}
