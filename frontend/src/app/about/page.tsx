export default function AboutPage() {
  return (
    <div className="container mx-auto max-w-2xl px-4 py-12">
      <h2 className="text-3xl font-bold text-blue-700 mb-6">About KeraTrack</h2>
      <div className="space-y-6 text-gray-700 text-lg">
        <div>
          <b>Academic Completion for DS Lab</b>
        </div>
        <ul className="list-disc list-inside text-base pl-2">
          <li><b>College:</b> A P Shah Institute of Technology, Thane</li>
          <li><b>Professor:</b> Ms. Shafaque</li>
          <li><b>Topic:</b> KeraTrack : Predict & Visualize Hair Loss Progression</li>
        </ul>
        <div>
          <b>Team Members:</b>
          <ul className="list-disc list-inside ml-4">
            <li>22104121 - Omkar Chandgaonkar</li>
            <li>22104089 - Paresh Gupta</li>
            <li>22104135 - Aditya Chaudhari</li>
            <li>22104063 - Sanika Ghorad</li>
          </ul>
        </div>
        <p>
          KeraTrack leverages deep learning to predict and visualize the progression of hair loss from scalp images. This tool empowers users to monitor their hair health in a quick, non-intrusive and accessible way.
        </p>
      </div>
    </div>
  );
}