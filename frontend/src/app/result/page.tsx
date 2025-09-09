"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

interface AnalysisResult {
  predicted_stage: keyof typeof stageDescriptions;
  confidence: number;
}

const stageDescriptions = {
  LEVEL_1: "Mild: Early signs of hair loss, usually not easily noticeable. Maintain a healthy diet and gentle hair care.",
  LEVEL_2: "Moderate: Visible thinning, especially at the front or crown. Consider consulting a dermatologist for personalized advice.",
  LEVEL_3: "Advanced: Significant thinning/bald patches. Medical intervention may be beneficial.",
  LEVEL_4: "Severe: Extensive hair loss, often leading to baldness. Professional treatment options should be explored.",
  LEVEL_5: "Complete: Total hair loss on the scalp. Options include wigs, hairpieces, or surgical solutions.",
};

export default function ResultPage() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const res = sessionStorage.getItem("keratrack-result");
    const img = sessionStorage.getItem("keratrack-image");
    if (!res || !img) {
      router.push("/");
      return;
    }
    setResult(JSON.parse(res));
    setImage(img);
  }, []);

  if (!result || !image) return null;

  return (
    <div className="container mx-auto max-w-2xl px-4 py-12">
      <h2 className="text-3xl font-bold text-blue-700 mb-8">Your Analysis Result</h2>
      <div className="bg-white rounded-lg shadow p-6 flex flex-col md:flex-row gap-8">
        <img src={image} alt="Uploaded scalp" className="w-48 h-48 object-cover rounded border" />
        <div className="flex-1 flex flex-col gap-3">
          <div>
            <span className="text-gray-700 font-medium">Result:</span>
            <span className="ml-2 text-2xl font-extrabold text-blue-700">{result.predicted_stage}</span>
          </div>
          <div>
            <span className="text-gray-700 font-medium">Confidence:</span>
            <span className="ml-2 text-blue-600 font-semibold">{(result.confidence * 100).toFixed(2)}%</span>
          </div>
          <div className="mt-3 text-gray-800 bg-blue-50 rounded p-3">
            <b>What does this mean?</b><br />
            {stageDescriptions[result.predicted_stage] || "No description available."}
          </div>
          <div className="mt-3">
            <b>Next Steps:</b>
            <ul className="list-disc ml-6 text-gray-700">
              <li>Consider consulting a dermatologist for a professional opinion.</li>
              <li>Track your progression with KeraTrack over time for better insights.</li>
              <li>Maintain a healthy lifestyle and follow good hair care practices.</li>
            </ul>
          </div>
          <div className="mt-4 flex gap-3">
            <Link href="/predict">
              <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-800 transition">Try Another</button>
            </Link>
            <Link href="/history">
              <button className="border border-blue-600 text-blue-700 px-4 py-2 rounded hover:bg-blue-50 transition">See History</button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}