"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface NutrientInfo {
  daily_target: number;
  unit: string;
  sources: string[];
  importance: string;
}

interface DietRecommendation {
  id: number;
  recommended_nutrients: Record<string, NutrientInfo>;
  meal_plan: Record<string, any>;
  supplements: Array<{
    name: string;
    dosage: string;
    reason: string;
    timing: string;
  }>;
  confidence_score: number;
  reasoning: string;
  created_at: string;
}

interface DietAssessment {
  age: number;
  gender: string;
  weight: number;
  height: number;
  activity_level: string;
  dietary_restrictions: string[];
  current_diet_pattern: string;
  health_conditions: string[];
  medications: string[];
}

export default function DietDashboard() {
  const [recommendations, setRecommendations] = useState<DietRecommendation | null>(null);
  const [hasAssessment, setHasAssessment] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showAssessmentForm, setShowAssessmentForm] = useState(false);
  const [assessment, setAssessment] = useState<DietAssessment>({
    age: 30,
    gender: "female",
    weight: 65,
    height: 165,
    activity_level: "moderate",
    dietary_restrictions: [],
    current_diet_pattern: "standard",
    health_conditions: [],
    medications: []
  });
  const router = useRouter();

  useEffect(() => {
    fetchRecommendations();
  }, []);

  const fetchRecommendations = async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/diet/recommendations", {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (response.status === 400) {
        // No assessment exists
        setHasAssessment(false);
        setShowAssessmentForm(true);
      } else if (response.ok) {
        const data = await response.json();
        setRecommendations(data);
        setHasAssessment(true);
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    } finally {
      setLoading(false);
    }
  };

  const submitAssessment = async (e: React.FormEvent) => {
    e.preventDefault();
    const token = localStorage.getItem("token");
    
    try {
      // Submit assessment
      const assessmentResponse = await fetch("http://localhost:8000/diet/assessment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(assessment)
      });

      if (assessmentResponse.ok) {
        // Now fetch recommendations
        await fetchRecommendations();
        setShowAssessmentForm(false);
      }
    } catch (error) {
      console.error("Error submitting assessment:", error);
    }
  };

  const handleCheckboxChange = (field: keyof DietAssessment, value: string) => {
    setAssessment(prev => {
      const currentValue = prev[field];
      if (Array.isArray(currentValue)) {
        return {
          ...prev,
          [field]: currentValue.includes(value)
            ? currentValue.filter(item => item !== value)
            : [...currentValue, value]
        };
      }
      return prev;
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center">
        <div className="text-blue-600 text-xl">Loading your personalized diet plan...</div>
      </div>
    );
  }

  if (showAssessmentForm) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 py-8">
        <div className="container mx-auto px-4 max-w-2xl">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h1 className="text-3xl font-bold text-blue-700 mb-6 text-center">
              🥗 Hair Health Diet Assessment
            </h1>
            <p className="text-gray-600 mb-8 text-center">
              Help us create your personalized nutrition plan for optimal hair health
            </p>

            <form onSubmit={submitAssessment} className="space-y-6">
              {/* Basic Info */}
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Age</label>
                  <input
                    type="number"
                    value={assessment.age}
                    onChange={(e) => setAssessment({...assessment, age: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Gender</label>
                  <select
                    value={assessment.gender}
                    onChange={(e) => setAssessment({...assessment, gender: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                  </select>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Weight (kg)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={assessment.weight}
                    onChange={(e) => setAssessment({...assessment, weight: parseFloat(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Height (cm)</label>
                  <input
                    type="number"
                    value={assessment.height}
                    onChange={(e) => setAssessment({...assessment, height: parseFloat(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Activity Level</label>
                <select
                  value={assessment.activity_level}
                  onChange={(e) => setAssessment({...assessment, activity_level: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="sedentary">Sedentary (little/no exercise)</option>
                  <option value="moderate">Moderate (exercise 3-4x/week)</option>
                  <option value="active">Active (exercise 5+ times/week)</option>
                </select>
              </div>

              {/* Dietary Restrictions */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Dietary Restrictions</label>
                <div className="grid md:grid-cols-3 gap-2">
                  {["vegetarian", "vegan", "gluten_free", "dairy_free", "nut_free", "soy_free"].map(restriction => (
                    <label key={restriction} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={assessment.dietary_restrictions.includes(restriction)}
                        onChange={() => handleCheckboxChange("dietary_restrictions", restriction)}
                        className="mr-2"
                      />
                      <span className="capitalize">{restriction.replace("_", " ")}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Health Conditions */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Health Conditions</label>
                <div className="grid md:grid-cols-3 gap-2">
                  {["thyroid", "pcos", "diabetes", "anemia", "autoimmune", "digestive_issues"].map(condition => (
                    <label key={condition} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={assessment.health_conditions.includes(condition)}
                        onChange={() => handleCheckboxChange("health_conditions", condition)}
                        className="mr-2"
                      />
                      <span className="capitalize">{condition.replace("_", " ")}</span>
                    </label>
                  ))}
                </div>
              </div>

              <button
                type="submit"
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 transition duration-300"
              >
                Generate My Personalized Diet Plan
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  if (!recommendations) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center">
        <div className="text-red-600 text-xl">Unable to load recommendations</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-blue-700 mb-2">
            🥗 Your Personalized Hair Health Diet Plan
          </h1>
          <p className="text-gray-600">
            AI-powered nutrition recommendations based on your hair analysis
          </p>
          <div className="mt-4 inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-full">
            <span className="mr-2">🎯</span>
            Confidence Score: {(recommendations.confidence_score * 100).toFixed(0)}%
          </div>
        </div>

        {/* AI Reasoning */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            🧠 Why This Plan is Perfect for You
          </h2>
          <p className="text-gray-600 leading-relaxed">
            {recommendations.reasoning}
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Nutrients */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              🎯 Your Daily Nutrient Targets
            </h2>
            <div className="space-y-4">
              {Object.entries(recommendations.recommended_nutrients).map(([nutrient, info]) => (
                <div key={nutrient} className="border-b border-gray-100 pb-4">
                  <div className="flex justify-between items-start mb-2">
                    <span className="font-medium text-gray-800 capitalize">
                      {nutrient.replace("_", " ")}
                    </span>
                    <span className="text-blue-600 font-semibold">
                      {info.daily_target} {info.unit}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{info.importance}</p>
                  <div className="flex flex-wrap gap-1">
                    {info.sources.slice(0, 3).map((source, idx) => (
                      <span key={idx} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                        {source}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Supplements */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              💊 Recommended Supplements
            </h2>
            <div className="space-y-4">
              {recommendations.supplements.map((supplement, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-800 mb-2">
                    {supplement.name}
                  </h3>
                  <div className="space-y-1 text-sm">
                    <p><span className="font-medium">Dosage:</span> {supplement.dosage}</p>
                    <p><span className="font-medium">Timing:</span> {supplement.timing}</p>
                    <p className="text-gray-600">{supplement.reason}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Meal Plan Preview */}
        <div className="bg-white rounded-xl shadow-lg p-6 mt-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">
            🍽️ Sample Meal Plan (Day 1)
          </h2>
          {recommendations.meal_plan.day_1 && (
            <div className="grid md:grid-cols-3 gap-6">
              {["breakfast", "lunch", "dinner"].map(meal => {
                const mealData = recommendations.meal_plan.day_1[meal];
                return mealData && (
                  <div key={meal} className="border border-gray-200 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-800 capitalize mb-2">{meal}</h3>
                    <h4 className="text-blue-600 font-medium mb-2">{mealData.name}</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      {mealData.ingredients?.map((ingredient: string, idx: number) => (
                        <li key={idx}>• {ingredient}</li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex justify-center space-x-4 mt-8">
          <button
            onClick={() => setShowAssessmentForm(true)}
            className="bg-gray-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-gray-700 transition duration-300"
          >
            Update Assessment
          </button>
          <button
            onClick={() => router.push("/diet/lifestyle")}
            className="bg-green-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-green-700 transition duration-300"
          >
            Track Lifestyle
          </button>
          <button
            onClick={() => router.push("/")}
            className="bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 transition duration-300"
          >
            Back to Hair Analysis
          </button>
        </div>
      </div>
    </div>
  );
}