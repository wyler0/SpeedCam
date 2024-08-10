import Image from "next/image";
import { SpeedCam } from "@/components/speed-cam";


export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <SpeedCam />
    </main>
  );
}
