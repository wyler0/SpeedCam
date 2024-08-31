// © 2024 Wyler Zahm. All rights reserved.

import { Toaster } from 'react-hot-toast';

import { SpeedCam } from "@/components/custom/speed-cam";


export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4">
      <SpeedCam />
      <Toaster position="bottom-right" />
    </main>
  );
}
