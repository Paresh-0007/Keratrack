import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "KeraTrack",
  description: "Predict & Visualize Hair Loss Progression",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 min-h-screen flex flex-col">
        {/* Navbar */}
        <nav className="bg-white shadow px-4 py-3 sticky top-0 z-20">
          <div className="container mx-auto flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 font-bold text-blue-600 text-xl">
              <span className="tracking-tight">KeraTrack</span>
            </Link>
            <div className="flex gap-6 text-sm">
              <Link className="hover:text-blue-600 transition" href="/">Home</Link>
              <Link className="hover:text-blue-600 transition" href="/predict">Predict</Link>
              <Link className="hover:text-blue-600 transition" href="/history">History</Link>
              <Link className="hover:text-blue-600 transition" href="/about">About Us</Link>
              <Link className="hover:text-blue-600 transition" href="/login">Login</Link>
              <Link className="hover:text-blue-600 transition" href="/signup">
                <span className="border px-3 py-1 rounded text-blue-600 border-blue-600 hover:bg-blue-50 transition">Sign Up</span>
              </Link>
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1">{children}</main>

        {/* Footer */}
        <footer className="bg-white border-t mt-10">
          <div className="container mx-auto px-4 py-6 flex flex-col md:flex-row justify-between items-center text-gray-500 text-sm">
            <div>
              &copy; {new Date().getFullYear()} KeraTrack &mdash; Academic Project, APSIT
            </div>
            {/* <div>
              Made with <span className="text-blue-600">♥</span> by Paresh Gupta
            </div> */}
          </div>
        </footer>
      </body>
    </html>
  );
}