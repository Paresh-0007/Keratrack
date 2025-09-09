"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function SignupPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const router = useRouter();

  const handleSignup = async (e:any) => {
    e.preventDefault();
    setErr("");
    try {
      const res = await fetch("http://localhost:8000/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
      });
      if (!res.ok) throw new Error("Signup failed");
      router.push("/login");
    } catch (e) {
      setErr("Error creating account");
    }
  };

  return (
    <div className="flex justify-center py-14">
      <div className="bg-white rounded-lg shadow max-w-md w-full p-8">
        <h2 className="text-2xl font-bold text-blue-700 mb-4">Sign Up</h2>
        <form onSubmit={handleSignup} className="flex flex-col gap-4">
          <input
            type="text"
            placeholder="Name"
            className="border rounded px-3 py-2"
            value={name}
            onChange={e => setName(e.target.value)}
            required
          />
          <input
            type="email"
            placeholder="Email"
            className="border rounded px-3 py-2"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            className="border rounded px-3 py-2"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
          <button className="bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition">
            Sign Up
          </button>
          {err && <p className="text-red-500 text-sm">{err}</p>}
        </form>
        <div className="mt-4 text-sm text-gray-600">
          Already have an account?{" "}
          <Link href="/login" className="underline text-blue-600 hover:text-blue-800">Login</Link>
        </div>
      </div>
    </div>
  );
}