export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const api = {
  getAttention: (wellId) => request(`/api/attention/${wellId}`),
  postWhatIf: (body) => request("/api/whatif", { method: "POST", body: JSON.stringify(body) }),
  postControlRun: (body) => request("/api/control/run", { method: "POST", body: JSON.stringify(body) }),
  getExperiments: () => request("/api/control/experiments"),
  getExperiment: (id) => request(`/api/control/experiments/${id}`),
  postDoseResponse: (body) => request("/api/dose_response/run", { method: "POST", body: JSON.stringify(body) }),
  postCsvAnalyze: async (file) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_BASE}/api/csv/analyze`, { method: "POST", body: form });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  },
};
