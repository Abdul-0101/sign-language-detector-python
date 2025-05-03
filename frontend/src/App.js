
import React, { useRef, useState, useEffect } from "react";
import axios from "axios";
export default function App() {
  const videoRef = useRef();
  const [letter, setLetter] = useState("");
  const [word, setWord] = useState("");

  const captureAndSend = async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg"));
    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    const resp = await axios.post("https://<YOUR-RENDER‑BACKEND>/infer/", form);
    setLetter(resp.data.letter);
    setWord(resp.data.word);
  };

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { videoRef.current.srcObject = stream; });
    const interval = setInterval(captureAndSend, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4">
      <video ref={videoRef} width="640" height="480" autoPlay />
      <div className="mt-4 text-xl">Letter: {letter}</div>
      <div className="text-2xl font-bold">Word: {word}</div>
    </div>
  );
}
