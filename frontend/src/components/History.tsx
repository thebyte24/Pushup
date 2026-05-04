import { useEffect, useState } from "react";

interface Session {
  Date: string;
  Total_Reps: string;
  Duration_Seconds: string;
}

export default function History() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = () => {
    fetch("http://localhost:8000/history")
      .then((r) => r.json())
      .then((data) => { setSessions(data); setLoading(false); })
      .catch(() => setLoading(false));
  };

  useEffect(() => { fetchHistory(); }, []);

  const deleteSession = (date: string) => {
    if (!window.confirm(`Delete session from ${date}?`)) return;
    fetch(`http://localhost:8000/history/${encodeURIComponent(date)}`, { method: "DELETE" })
      .then(() => fetchHistory());
  };

  if (loading) return <p className="loading">Loading history...</p>;
  if (!sessions.length) return <p className="loading">No sessions yet. Start a workout!</p>;

  const totalReps = sessions.reduce((s, r) => s + parseInt(r.Total_Reps), 0);

  return (
    <div className="history-page">
      <div className="history-summary">
        <div className="summary-card">
          <span>{sessions.length}</span>
          <label>Sessions</label>
        </div>
        <div className="summary-card">
          <span>{totalReps}</span>
          <label>Total Reps</label>
        </div>
      </div>

      <table className="history-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Reps</th>
            <th>Duration</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {[...sessions].reverse().map((s, i) => (
            <tr key={i}>
              <td>{s.Date}</td>
              <td>{s.Total_Reps}</td>
              <td>{parseFloat(s.Duration_Seconds).toFixed(1)}s</td>
              <td>
                <button className="delete-btn" onClick={() => deleteSession(s.Date)}>🗑</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
