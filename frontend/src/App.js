import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const App = () => {
  const webcamRef = useRef(null);
  const [predictedLetter, setPredictedLetter] = useState("");
  const [predictedWord, setPredictedWord] = useState("");

  const captureFrame = async () => {
    if (webcamRef.current) {
      const screenshot = webcamRef.current.getScreenshot();
      if (screenshot) {
        const blob = await fetch(screenshot).then(res => res.blob());
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
          const response = await axios.post("http://localhost:8000/predict/", formData, {
            headers: { "Content-Type": "multipart/form-data" },
          });
          setPredictedLetter(response.data.letter || "");
          setPredictedWord(response.data.word || "");
        } catch (error) {
          console.error("Prediction failed:", error);
        }
      }
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      captureFrame();
    }, 1000); // Every second
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h1>Sign Language Detection</h1>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={400}
        height={300}
        style={{ border: "2px solid black" }}
      />
      <div style={{ marginTop: "20px" }}>
        <h2>Predicted Letter: {predictedLetter}</h2>
        <h2>Predicted Word: {predictedWord}</h2>
      </div>
    </div>
  );
};

export default App;
