"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function Landing() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  // For Drag and Drop
  const [dragActive, setDragActive] = useState(false);

  const handleUpload = async (e:any) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const token = localStorage.getItem("token");
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
    if (res.ok) {
      const result = await res.json();
      // Save result and image in sessionStorage, then redirect
      sessionStorage.setItem("keratrack-result", JSON.stringify(result));
      // Save uploaded file as base64 for display
      const reader = new FileReader();
      reader.onloadend = () => {
        if (reader.result && typeof reader.result === 'string') {
          sessionStorage.setItem("keratrack-image", reader.result);
          router.push("/result");
        }
      };
      reader.readAsDataURL(file);
    }
    setLoading(false);
  };

  return (
    <>
      {/* HERO */}
      <section className="bg-gradient-to-r from-blue-50 to-blue-100 py-16">
        <div className="container mx-auto px-4 flex flex-col md:flex-row items-center justify-between">
          <div className="flex-1">
            <h1 className="text-4xl md:text-5xl font-extrabold text-blue-700 mb-4">Understand Your Hair Loss Stage Instantly</h1>
            <p className="text-gray-700 text-lg mb-6">
              KeraTrack uses AI to analyze your scalp image and help you take control of your hair health. Early detection is key!
            </p>
            <ul className="mb-6 space-y-1 text-blue-600 font-medium">
              <li>✓ Accurate Stage Classification</li>
              <li>✓ Personalized Hair Care Insights</li>
              <li>✓ Easy Progress Tracking</li>
            </ul>
            <a href="#upload" className="inline-block bg-blue-600 text-white px-6 py-3 rounded shadow hover:bg-blue-700 transition">
              Try Now (Free)
            </a>
          </div>
          <div className="flex-1 flex justify-center mt-10 md:mt-0">
            <img src="/hair-loss-illustration.svg" className="w-80 h-80 object-contain" alt="KeraTrack Illustration" />
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="container mx-auto px-4 py-14">
        <h2 className="text-2xl font-bold text-center text-blue-700 mb-8">How It Works</h2>
        <div className="flex flex-col md:flex-row justify-center gap-8">
          <StepCard number={1} text="Upload Your Scalp Image" icon="📤" />
          <StepCard number={2} text="AI Analyzes Hair Loss" icon="🤖" />
          <StepCard number={3} text="See Your Hair Loss Stage" icon="📊" />
        </div>
      </section>

      {/* FEATURES SECTION */}
      <section className="bg-gradient-to-r from-green-50 to-blue-50 py-16">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-12">
            Complete Hair Health Solution
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 text-center">
              <div className="text-4xl mb-4">🥗</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">AI Diet Recommendations</h3>
              <p className="text-gray-600 mb-4">
                Get personalized nutrition plans based on your hair loss stage and health profile
              </p>
              <Link
                href="/diet"
                className="inline-block bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition"
              >
                Get Diet Plan
              </Link>
            </div>
            
            <div className="bg-white rounded-xl shadow-lg p-6 text-center">
              <div className="text-4xl mb-4">📊</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Lifestyle Tracking</h3>
              <p className="text-gray-600 mb-4">
                Monitor stress, sleep, and exercise to understand their impact on your hair health
              </p>
              <Link
                href="/diet/lifestyle"
                className="inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
              >
                Start Tracking
              </Link>
            </div>
            
            <div className="bg-white rounded-xl shadow-lg p-6 text-center">
              <div className="text-4xl mb-4">📈</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Progress Analytics</h3>
              <p className="text-gray-600 mb-4">
                Track your hair health journey with detailed analytics and correlations
              </p>
              <Link
                href="/history"
                className="inline-block bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition"
              >
                View Progress
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* BENEFITS */}
      <section className="bg-blue-50 py-12 mt-10">
        <div className="container mx-auto px-4 flex flex-col md:flex-row items-center gap-10">
          <div className="flex-1">
            <h3 className="text-xl font-bold text-blue-700 mb-3">Why KeraTrack?</h3>
            <ul className="space-y-2 text-gray-700">
              <li><span className="font-bold text-blue-600">Fast & Private:</span> Instant results, no data shared with third parties.</li>
              <li><span className="font-bold text-blue-600">Track Progress:</span> See how your hair health changes over time.</li>
              <li><span className="font-bold text-blue-600">Science-Backed:</span> Developed by engineering students with clinical research in mind.</li>
            </ul>
          </div>
          <div className="flex-1 flex justify-center">
            <img src="/progress-chart-demo.png" className="w-72 h-40 object-contain" alt="Progress Chart Illustration" />
          </div>
        </div>
      </section>
    </>
  );
}

// Helper: Visual step card
function StepCard({ number, text, icon }: { number: number; text: string; icon: string }) {
  return (
    <div className="flex flex-col items-center bg-white rounded-lg shadow px-6 py-6 w-64">
      <div className="text-4xl mb-4">{icon}</div>
      <div className="text-blue-700 font-bold text-lg mb-1">Step {number}</div>
      <div className="text-gray-700 text-center">{text}</div>
    </div>
  );
}