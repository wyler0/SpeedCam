// © 2024 Wyler Zahm. All rights reserved.

import { useState, useEffect } from 'react';
import Image from 'next/image';

import { getEndpoint } from '@/api/endpoints';

export function LatestDetectionImage() {
  const [imageUrl, setImageUrl] = useState<string | null>(getEndpoint('LATEST_IMAGE_URL'));
  const [timestamp, setTimestamp] = useState<number>(Date.now());

  useEffect(() => {
    const fetchLatestImage = async () => {
      setImageUrl(`${getEndpoint('LATEST_IMAGE_URL')}?t=${Date.now()}`);
      setTimestamp(Date.now());
      // try {
      //   const response = await fetch(getEndpoint('LATEST_IMAGE_STATUS'));
      //   const data = await response.json();
      //   // if (data.has_new_image) {
      //   // }
      // } catch (error) {
      //   console.error('Error fetching latest image status:', error);
      // }
    };

    fetchLatestImage();
    const intervalId = setInterval(fetchLatestImage, 1000);

    return () => clearInterval(intervalId);
  }, []);

  if (!imageUrl) {
    return <div>No new image available</div>;
  }

  return (
    <div className="mt-4 text-center">
      <Image
        src={imageUrl}
        alt="Latest detection"
        width={640}
        height={480}
        className="rounded-lg shadow-md mx-auto"
        unoptimized={true}
      />
      <h3 className="text-lg font-semibold my-2">Latest Detection Image</h3>
    </div>
  );
}