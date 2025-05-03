import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./App.css";

export default function App() {
  const webcamRef = useRef();
  const [letter, setLetter] = useState("");
  const [word, setWord] = useState("");

  // capture & send
  const capture = async () => {
    if (!webcamRef.current) return;
    const img = webcamRef.current.getScreenshot();
    const blob = await fetch(img).then(r => r.blob());
    const fd = new FormData();
    fd.append("file", blob, "frame.jpg");

    try {
      const res = await axios.post(
        "https://sign-language-web.onrender.com/predict/",
        fd,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setLetter(res.data.letter);
      setWord(res.data.word);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    const id = setInterval(capture, 500);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="container">
      <div className="video-panel">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="video"
        />
        <div className="overlay">
          <div className="detected">Detected: {letter || "-"}</div>
          {letter && <div className="remove">Remove hand to detect next letter</div>}
        </div>
      </div>
      <div className="output-panel">
        <h3>Current Word: {word.charAt(0) || "-"}</h3>
        <h3>Predicted Word: {word || "-"}</h3>
      </div>
    </div>
  );
}
