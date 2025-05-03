import React, { useRef, useEffect, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const [prediction, setPrediction] = useState({ letter: "", word: "" });

  useEffect(() => {
    const startCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    };
    startCamera();
  }, []);

  useEffect(() => {
    const interval = setInterval(captureAndSendFrame, 1000); // Adjust frequency here
    return () => clearInterval(interval);
  }, []);

  const captureAndSendFrame = async () => {
    const video = videoRef.current;
    if (!video) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL("image/jpeg");

    try {
      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: dataURL }),
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error("Error sending frame:", error);
    }
  };

  return (
    <div className="App">
      <h1>Sign Language Detector</h1>
      <video ref={videoRef} autoPlay playsInline className="video" />
      <div className="output">
        <h2>Letter: {prediction.letter || "-"}</h2>
        <h2>Word: {prediction.word || "-"}</h2>
      </div>
    </div>
  );
}

export default App;
