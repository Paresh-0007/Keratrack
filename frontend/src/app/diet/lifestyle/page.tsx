"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface LifestyleEntry {
  id?: number;
  date: string;
  stress_level: number;
  sleep_hours: number;
  exercise_minutes: number;
  water_intake: number;
  notes: string;
}

export default function LifestyleTracker() {
  const [entry, setEntry] = useState<LifestyleEntry>({
    date: new Date().toISOString().split('T')[0],
    stress_level: 5,
    sleep_hours: 7,
    exercise_minutes: 30,
    water_intake: 2.0,
    notes: ""
  });
  const [history, setHistory] = useState<LifestyleEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/diet/lifestyle/history?days=30", {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      }
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  const submitEntry = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    const token = localStorage.getItem("token");

    try {
      const response = await fetch("http://localhost:8000/diet/lifestyle", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(entry)
      });

      if (response.ok) {
        await fetchHistory();
        // Reset form with today's date
        setEntry({
          date: new Date().toISOString().split('T')[0],
          stress_level: 5,
          sleep_hours: 7,
          exercise_minutes: 30,
          water_intake: 2.0,
          notes: ""
        });
        alert("Lifestyle entry logged successfully!");
      }
    } catch (error) {
      console.error("Error submitting entry:", error);
      alert("Error logging entry. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getStressLabel = (level: number) => {
    if (level <= 3) return { label: "Low", color: "text-green-600", bg: "bg-green-100" };
    if (level <= 6) return { label: "Moderate", color: "text-yellow-600", bg: "bg-yellow-100" };
    return { label: "High", color: "text-red-600", bg: "bg-red-100" };
  };

  const getSleepQuality = (hours: number) => {
    if (hours >= 7 && hours <= 9) return { label: "Optimal", color: "text-green-600" };
    if (hours >= 6 && hours < 7) return { label: "Good", color: "text-yellow-600" };
    return { label: "Poor", color: "text-red-600" };
  };

  const getHydrationStatus = (liters: number) => {
    if (liters >= 2.5) return { label: "Excellent", color: "text-blue-600" };
    if (liters >= 2) return { label: "Good", color: "text-green-600" };
    return { label: "Low", color: "text-red-600" };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 py-8">
      <div className="container mx-auto px-4 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-green-700 mb-2">
            📊 Lifestyle Tracking for Hair Health
          </h1>
          <p className="text-gray-600">
            Track daily factors that affect your hair health and see correlations with your progress
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Entry Form */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              📝 Log Today's Data
            </h2>

            <form onSubmit={submitEntry} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Date</label>
                <input
                  type="date"
                  value={entry.date}
                  onChange={(e) => setEntry({...entry, date: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Stress Level: {entry.stress_level}/10
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={entry.stress_level}
                  onChange={(e) => setEntry({...entry, stress_level: parseInt(e.target.value)})}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Very Low</span>
                  <span>Moderate</span>
                  <span>Very High</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sleep Hours</label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="12"
                  value={entry.sleep_hours}
                  onChange={(e) => setEntry({...entry, sleep_hours: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Exercise (minutes)</label>
                <input
                  type="number"
                  min="0"
                  value={entry.exercise_minutes}
                  onChange={(e) => setEntry({...entry, exercise_minutes: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Water Intake (liters)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={entry.water_intake}
                  onChange={(e) => setEntry({...entry, water_intake: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Notes (optional)</label>
                <textarea
                  value={entry.notes}
                  onChange={(e) => setEntry({...entry, notes: e.target.value})}
                  placeholder="Any observations about your hair, mood, diet, etc."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-green-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-green-700 transition duration-300 disabled:opacity-50"
              >
                {loading ? "Logging..." : "Log Entry"}
              </button>
            </form>
          </div>

          {/* Recent History */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              📈 Recent Entries
            </h2>

            {history.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                No entries yet. Start tracking your lifestyle factors!
              </p>
            ) : (
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {history.slice(0, 10).map((historyEntry) => {
                  const stressInfo = getStressLabel(historyEntry.stress_level);
                  const sleepInfo = getSleepQuality(historyEntry.sleep_hours);
                  const hydrationInfo = getHydrationStatus(historyEntry.water_intake);

                  return (
                    <div key={historyEntry.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <span className="font-medium text-gray-800">
                          {new Date(historyEntry.date).toLocaleDateString()}
                        </span>
                        <div className="flex space-x-2">
                          <span className={`px-2 py-1 rounded text-xs ${stressInfo.bg} ${stressInfo.color}`}>
                            Stress: {stressInfo.label}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Sleep:</span>
                          <span className={`ml-1 ${sleepInfo.color}`}>
                            {historyEntry.sleep_hours}h ({sleepInfo.label})
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">Exercise:</span>
                          <span className="ml-1">{historyEntry.exercise_minutes}min</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Water:</span>
                          <span className={`ml-1 ${hydrationInfo.color}`}>
                            {historyEntry.water_intake}L ({hydrationInfo.label})
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">Stress:</span>
                          <span className={`ml-1 ${stressInfo.color}`}>
                            {historyEntry.stress_level}/10
                          </span>
                        </div>
                      </div>

                      {historyEntry.notes && (
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          <p className="text-sm text-gray-600 italic">"{historyEntry.notes}"</p>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Quick Stats */}
        {history.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-6 mt-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              📊 Your 30-Day Averages
            </h2>
            <div className="grid md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {(history.reduce((sum, entry) => sum + entry.stress_level, 0) / history.length).toFixed(1)}
                </div>
                <div className="text-sm text-gray-600">Avg Stress Level</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  {(history.reduce((sum, entry) => sum + entry.sleep_hours, 0) / history.length).toFixed(1)}h
                </div>
                <div className="text-sm text-gray-600">Avg Sleep</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">
                  {Math.round(history.reduce((sum, entry) => sum + entry.exercise_minutes, 0) / history.length)}min
                </div>
                <div className="text-sm text-gray-600">Avg Exercise</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {(history.reduce((sum, entry) => sum + entry.water_intake, 0) / history.length).toFixed(1)}L
                </div>
                <div className="text-sm text-gray-600">Avg Water</div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex justify-center space-x-4 mt-8">
          <button
            onClick={() => router.push("/diet")}
            className="bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 transition duration-300"
          >
            View Diet Plan
          </button>
          <button
            onClick={() => router.push("/")}
            className="bg-gray-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-gray-700 transition duration-300"
          >
            Back to Hair Analysis
          </button>
        </div>
      </div>
    </div>
  );
}