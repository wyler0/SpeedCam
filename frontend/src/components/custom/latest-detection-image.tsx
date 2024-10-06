// © 2024 Wyler Zahm. All rights reserved.

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Image from 'next/image';
import { getEndpoint } from '@/api/endpoints';

interface CropLine {
  position: number;
  isTop: boolean;
  isLeft: boolean;
}

interface LatestDetectionImageProps {
  allowCropAdjustment?: boolean;
  onCropChange?: (cropValues: {
    left_crop_l2r: number;
    right_crop_l2r: number;
    left_crop_r2l: number;
    right_crop_r2l: number;
  }) => void;
  cropValues?: {
    left_crop_l2r: number;
    right_crop_l2r: number;
    left_crop_r2l: number;
    right_crop_r2l: number;
  };
}

const DEFAULT_CROP_VALUES = {
  left_crop_l2r: 0,
  right_crop_l2r: 100,
  left_crop_r2l: 0,
  right_crop_r2l: 100,
};

export function LatestDetectionImage({
  allowCropAdjustment = false,
  onCropChange,
  cropValues = DEFAULT_CROP_VALUES,
}: LatestDetectionImageProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(getEndpoint('LATEST_IMAGE_URL'));
  const [timestamp, setTimestamp] = useState<number>(Date.now());
  const [cropLines, setCropLines] = useState<CropLine[]>([
    { position: cropValues.left_crop_l2r, isTop: true, isLeft: true },
    { position: cropValues.right_crop_l2r, isTop: true, isLeft: false },
    { position: cropValues.left_crop_r2l, isTop: false, isLeft: true },
    { position: cropValues.right_crop_r2l, isTop: false, isLeft: false },
  ]);

  const imageRef = useRef<HTMLImageElement>(null);
  const isDraggingRef = useRef(false);
  const activeCropLineRef = useRef<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 640, height: 480 });
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    const fetchLatestImage = async () => {
      setImageUrl(`${getEndpoint('LATEST_IMAGE_URL')}?t=${Date.now()}`);
      setTimestamp(Date.now());
    };

    fetchLatestImage();
    const intervalId = setInterval(fetchLatestImage, 1000);

    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    setCropLines([
      { position: cropValues.left_crop_l2r, isTop: true, isLeft: true },
      { position: cropValues.right_crop_l2r, isTop: true, isLeft: false },
      { position: cropValues.left_crop_r2l, isTop: false, isLeft: true },
      { position: cropValues.right_crop_r2l, isTop: false, isLeft: false },
    ]);
  }, [cropValues]);

  const updateImageDimensions = useCallback(() => {
    if (imageRef.current && containerRef.current) {
      const containerWidth = containerRef.current.clientWidth;
      const naturalWidth = imageRef.current.naturalWidth || 640;
      const naturalHeight = imageRef.current.naturalHeight || 480;
      const imageAspectRatio = naturalWidth / naturalHeight;
      const imageWidth = Math.min(640, containerWidth);
      const imageHeight = Math.round(imageWidth / imageAspectRatio);
      setImageDimensions({ width: imageWidth, height: imageHeight });
    }
  }, []);

  useEffect(() => {
    window.addEventListener('resize', updateImageDimensions);
    return () => window.removeEventListener('resize', updateImageDimensions);
  }, [updateImageDimensions]);

  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
    updateImageDimensions();
  }, [updateImageDimensions]);

  const handleMouseDown = useCallback((index: number) => (e: React.MouseEvent) => {
    if (!allowCropAdjustment) return;
    e.preventDefault();
    isDraggingRef.current = true;
    activeCropLineRef.current = index;
  }, [allowCropAdjustment]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingRef.current || activeCropLineRef.current === null || !imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const newPosition = ((e.clientX - rect.left) / rect.width) * 100;
    const clampedPosition = Math.max(0, Math.min(100, newPosition));

    setCropLines(prevLines => 
      prevLines.map((line, index) => 
        index === activeCropLineRef.current ? { ...line, position: clampedPosition } : line
      )
    );
  }, []);

  const handleMouseUp = useCallback(() => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    if (onCropChange) {
      onCropChange({
        left_crop_l2r: cropLines[0].position,
        right_crop_l2r: cropLines[1].position,
        left_crop_r2l: cropLines[2].position,
        right_crop_r2l: cropLines[3].position,
      });
    }
  }, [cropLines, onCropChange]);

  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  const renderCropLines = () => {
    return cropLines.map((line, index) => (
      <div
        key={index}
        style={{
          position: 'absolute',
          top: line.isTop ? '0%' : '50%',
          left: `${(imageDimensions.width * line.position) / 100}px`,
          width: '2px',
          height: '50%',
          backgroundColor: 'red',
          cursor: allowCropAdjustment ? 'ew-resize' : 'default',
        }}
        onMouseDown={handleMouseDown(index)}
      />
    ));
  };

  if (!imageUrl) {
    return <div>No new image available</div>;
  }

  return (
    <div className="mt-4 text-center relative" ref={containerRef}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <Image
          ref={imageRef}
          src={imageUrl}
          alt="Latest detection"
          width={imageDimensions.width}
          height={imageDimensions.height}
          className="rounded-lg shadow-md mx-auto"
          unoptimized={true}
          onLoad={handleImageLoad}
        />
        {imageLoaded && renderCropLines()}
      </div>
      <h3 className="text-lg font-semibold my-2">Latest Detection Image</h3>
    </div>
  );
}