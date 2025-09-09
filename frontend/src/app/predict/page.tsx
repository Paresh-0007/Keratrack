"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

interface PredictionResult {
  id: number;
  predicted_stage: string;
  confidence: number;
}

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleFileChange = (e: any) => {
    setFile(e.target.files[0]);
    setResult(null);
    setErr("");
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    setErr("");
    setResult(null);
    if (!file) {
      setErr("Please select an image file.");
      return;
    }
    setLoading(true);
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        setErr("Please login first.");
        router.push("/login");
        return;
      }
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setErr("Prediction failed");
    }
    setLoading(false);
  };

  return (
    <div className="max-w-lg mx-auto mt-12 bg-white p-8 rounded shadow">
      <h2 className="text-2xl font-bold mb-4 text-blue-700">Upload Image</h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
          file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700
          hover:file:bg-blue-100"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
          disabled={loading}
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
        {err && <p className="text-red-500 text-sm">{err}</p>}
      </form>

      {result && (
        <div className="mt-8 bg-blue-50 p-4 rounded shadow flex flex-col gap-2">
          <h3 className="font-semibold mb-2">Prediction Result</h3>
          <p>
            <span className="font-medium">Stage:</span> {result.predicted_stage}
          </p>
          <p>
            <span className="font-medium">Confidence:</span>{" "}
            {(result.confidence * 100).toFixed(2)}%
          </p>
          <p className="text-gray-400 text-xs">Prediction ID: {result.id}</p>
        </div>
      )}
    </div>
  );
}
