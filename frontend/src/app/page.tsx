import { SpeedCam } from "@/components/custom/speed-cam";
import { Toaster } from 'react-hot-toast';


export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4">
      <SpeedCam />
      <Toaster position="bottom-right" />
    </main>
  );
}
