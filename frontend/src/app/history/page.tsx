"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Chart as ChartJS,
  LineElement,
  LinearScale,
  CategoryScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import Link from "next/link";

ChartJS.register(
  LineElement,
  LinearScale,
  CategoryScale,
  PointElement,
  Tooltip,
  Legend
);

const STAGE_MAP = {
  LEVEL_1: { value: 1, name: "Level 1", color: "#22c55e" },
  LEVEL_2: { value: 2, name: "Level 2", color: "#4ade80" },
  LEVEL_3: { value: 3, name: "Level 3", color: "#facc15" },
  LEVEL_4: { value: 4, name: "Level 4", color: "#fb923c" },
  LEVEL_5: { value: 5, name: "Level 5", color: "#f472b6" },
  LEVEL_6: { value: 6, name: "Level 6", color: "#a78bfa" },
  LEVEL_7: { value: 7, name: "Level 7", color: "#ef4444" },
};

const FILTERS = [
  { label: "Last 7 Days", value: "week" },
  { label: "Last 30 Days", value: "month" },
  { label: "All Time", value: "all" },
];

interface HistoryItem {
  id: string;
  created_at: string;
  predicted_stage: keyof typeof STAGE_MAP;
  confidence: number;
  image_path: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filtered, setFiltered] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [modalImg, setModalImg] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const fetchHistory = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        setError("Please login to view history.");
        router.push("/login");
        return;
      }
      try {
        const res = await fetch("http://localhost:8000/history", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!res.ok) throw new Error("Could not fetch history");
        const data = await res.json();
        setHistory(data);
        setFiltered(data);
      } catch (err) {
        setError("Failed to load history");
      }
      setLoading(false);
    };
    fetchHistory();
  }, []);

  useEffect(() => {
    if (filter === "all") {
      setFiltered(history);
    } else {
      const now = new Date();
      let cutoff = new Date();
      if (filter === "week") cutoff.setDate(now.getDate() - 7);
      if (filter === "month") cutoff.setDate(now.getDate() - 30);
      setFiltered(
        history.filter((h) => new Date(h.created_at) > cutoff)
      );
    }
  }, [filter, history]);

  const handleExport = () => {
    const csv =
      "Date,Stage,Confidence (%)\n" +
      filtered
        .map(
          (h) =>
            `${new Date(h.created_at).toLocaleString()},${h.predicted_stage},${(
              h.confidence * 100
            ).toFixed(2)}`
        )
        .join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "keratrack_history.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Chart Data
  const chartData = {
    labels: filtered.map((h) => new Date(h.created_at).toLocaleDateString()),
    datasets: [
      {
        label: "Hair Loss Stage (Lower is Better)",
        data: filtered.map((h) => STAGE_MAP[h.predicted_stage as keyof typeof STAGE_MAP]?.value || 0),
        fill: false,
        borderColor: "#2563eb",
        backgroundColor: "#2563eb",
        tension: 0.3,
        pointBackgroundColor: filtered.map(
          (h) => STAGE_MAP[h.predicted_stage as keyof typeof STAGE_MAP]?.color || "#ccc"
        ),
        pointBorderColor: filtered.map(
          (h) => STAGE_MAP[h.predicted_stage as keyof typeof STAGE_MAP]?.color || "#ccc"
        ),
        pointRadius: 6,
      },
    ],
  };

  const uniqueStages = [
    ...new Set(filtered.map((h) => STAGE_MAP[h.predicted_stage as keyof typeof STAGE_MAP]?.value)),
  ];
  const yMin = Math.min(...uniqueStages, 1) - 0.5;
  const yMax = Math.max(...uniqueStages, 7) + 0.5;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            const idx = context.dataIndex;
            const h = filtered[idx];
            return (
              STAGE_MAP[h.predicted_stage as keyof typeof STAGE_MAP ]?.name +
              ` (${(h.confidence * 100).toFixed(1)}%)`
            );
          },
        },
      },
    },
    scales: {
      y: {
        min: yMin,
        max: yMax,
        stepSize: 1,
        ticks: {
          callback: function (value:any) {
            return (
              Object.keys(STAGE_MAP).find(
                (key) => STAGE_MAP[key as keyof typeof STAGE_MAP].value === value
              ) || value
            );
          },
        },
      },
    },
  };

  // Optional: Delete entry handler (requires backend DELETE support)
  // const handleDelete = async (id) => {
  //   const token = localStorage.getItem("token");
  //   await fetch(`http://localhost:8000/history/${id}`, {
  //     method: "DELETE",
  //     headers: { Authorization: `Bearer ${token}` },
  //   });
  //   setHistory(history.filter((h) => h.id !== id));
  // };

  return (
    <div className="container mx-auto max-w-4xl px-4 py-12">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 gap-4">
        <h2 className="text-2xl font-bold text-blue-700">
          Your Hair Health Timeline
        </h2>
        <div className="flex flex-wrap gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.value}
              className={`px-3 py-1 rounded border text-sm ${
                filter === f.value
                  ? "bg-blue-600 border-blue-600 text-white"
                  : "border-blue-600 text-blue-600 hover:bg-blue-50"
              }`}
              onClick={() => setFilter(f.value)}
            >
              {f.label}
            </button>
          ))}
          <button
            className="border border-blue-600 text-blue-600 px-4 py-1 rounded hover:bg-blue-50 transition text-sm"
            onClick={handleExport}
            disabled={filtered.length === 0}
          >
            Export CSV
          </button>
          <Link href="/predict">
            <button className="bg-blue-600 text-white px-4 py-1 rounded hover:bg-blue-700 transition text-sm">
              + Add New Scan
            </button>
          </Link>
        </div>
      </div>
      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">{error}</p>}

      {filtered.length > 0 && (
        <>
          <div className="bg-white rounded shadow p-4 mb-8">
            <h3 className="font-semibold mb-2">Progression Chart</h3>
            <Line data={chartData} options={chartOptions} height={120} />
          </div>
          <div>
            <h3 className="font-semibold mb-2">Timeline</h3>
            <ul className="space-y-4">
              {filtered
                .slice()
                .reverse()
                .map((item) => (
                  <li
                    key={item.id}
                    className={`rounded-lg p-4 flex items-center gap-4 shadow border-l-8`}
                    style={{
                      borderColor: STAGE_MAP[item.predicted_stage]?.color || "#2563eb",
                      background: "#f1f5f9",
                    }}
                  >
                    <button
                      className="w-16 h-16 rounded border bg-white hover:scale-105 transition"
                      onClick={() =>
                        setModalImg(
                          `http://localhost:8000/${item.image_path.replace(/\\/g, "/")}`
                        )
                      }
                      aria-label="View full image"
                    >
                      <img
                        src={`http://localhost:8000/${item.image_path.replace(
                          /\\/g,
                          "/"
                        )}`}
                        alt="Prediction"
                        className="w-full h-full object-cover rounded"
                      />
                    </button>
                    <div>
                      <div
                        className="font-bold text-lg"
                        style={{
                          color: STAGE_MAP[item.predicted_stage]?.color || "#2563eb",
                        }}
                      >
                        {STAGE_MAP[item.predicted_stage]?.name || item.predicted_stage}
                      </div>
                      <div className="text-gray-700 text-sm">
                        {new Date(item.created_at).toLocaleString()}
                      </div>
                      <div className="text-gray-500 text-xs">
                        Confidence: {(item.confidence * 100).toFixed(2)}%
                      </div>
                    </div>
                    {/* <button
                      className="ml-auto text-xs text-red-600 hover:underline"
                      onClick={() => handleDelete(item.id)}
                    >
                      Delete
                    </button> */}
                  </li>
                ))}
            </ul>
          </div>
        </>
      )}
      {!loading && filtered.length === 0 && (
        <div className="flex flex-col items-center mt-20 text-center">
          <img src="/empty-history.svg" className="w-48 mb-4" alt="No history" />
          <p className="text-gray-600 font-medium">
            No scans found.<br />Start your hair health journey by uploading your first image!
          </p>
        </div>
      )}
      {/* Image Modal */}
      {modalImg && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setModalImg(null)}
        >
          <img
            src={modalImg}
            alt="Full Scan"
            className="max-w-full max-h-[80vh] rounded shadow-lg border-4 border-white"
            onClick={e => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}