import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/metadata`)
      .then((r) => r.json())
      .then(setMetadata)
      .catch(() => setMetadata({ input_dim: 8575 }));
  }, []);

  const onSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      if (!file) {
        setError("Wybierz plik z widmem spektralnym.");
        return;
      }
      setLoading(true);
      setError(null);
      setResult(null);
      const form = new FormData();
      form.append("file", file);
      try {
        const res = await fetch(`${API_BASE}/api/predict/upload`, {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        if (!res.ok) {
          const detail =
            typeof data.detail === "string"
              ? data.detail
              : JSON.stringify(data.detail);
          throw new Error(detail || "Błąd serwera");
        }
        setResult(data);
      } catch (err) {
        setError(err.message || "Nie udało się wykonać predykcji.");
      } finally {
        setLoading(false);
      }
    },
    [file],
  );

  const dim = metadata?.input_dim ?? 8575;

  return (
    <div className="page">
      <header className="hero">
        <h1>Parametry gwiazd z widma</h1>
        <p>
          Wrzuć widmo spektralne ({dim} punktów) lub plik FITS (mwmStar/APOGEE). Model oszacuje Teff, log&nbsp;g
          i [Fe/H].
        </p>
      </header>

      <form className="card upload-card" onSubmit={onSubmit}>
        <label className="dropzone">
          <input
            type="file"
            accept=".csv,.json,.npy,.h5,.hdf5,.fits,.fit,.fts"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <span className="dropzone-title">
            {file ? file.name : "Kliknij lub przeciągnij plik"}
          </span>
          <span className="dropzone-hint">
             FITS - jedno widmo na plik
          </span>
        </label>

        <button type="submit" disabled={loading || !file}>
          {loading ? "Obliczanie… (ok. 15 s)" : "Przewidź parametry"}
        </button>
      </form>

      {error && <div className="card error-card">{error}</div>}

      {result && (
        <section className="card results-card">
          <h2>Wynik</h2>
          <dl className="results-grid">
            {Object.entries(result.labels).map(([name, value]) => (
              <div key={name} className="result-item">
                <dt>{name}</dt>
                <dd>
                  {name === "Teff"
                    ? `${value.toFixed(0)} K`
                    : value.toFixed(3)}
                </dd>
              </div>
            ))}
          </dl>
          <p className="nll">
            Negatywna log-wiarygodność (NLL):{" "}
            <strong>{result.nll.toFixed(2)}</strong>
          </p>
        </section>
      )}

      <footer className="footer">
        Model: cINN (FrEIA) · dane treningowe: stellar_training_data.h5
      </footer>
    </div>
  );
}
