"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const router = useRouter();

  const handleLogin = async (e:any) => {
    e.preventDefault();
    setErr("");
    try {
      const form = new FormData();
      form.append("username", email);
      form.append("password", password);
      const res = await fetch("http://localhost:8000/token", {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error("Login failed");
      const data = await res.json();
      localStorage.setItem("token", data.access_token);
      router.push("/predict");
    } catch (e) {
      setErr("Invalid credentials");
    }
  };

  return (
    <div className="flex justify-center py-14">
      <div className="bg-white rounded-lg shadow max-w-md w-full p-8">
        <h2 className="text-2xl font-bold text-blue-700 mb-4">Login</h2>
        <form onSubmit={handleLogin} className="flex flex-col gap-4">
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
            Login
          </button>
          {err && <p className="text-red-500 text-sm">{err}</p>}
        </form>
        <div className="mt-4 text-sm text-gray-600">
          Don't have an account?{" "}
          <Link href="/signup" className="underline text-blue-600 hover:text-blue-800">Sign Up</Link>
        </div>
      </div>
    </div>
  );
}