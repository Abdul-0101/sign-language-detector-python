import React, { useRef, useState, useEffect } from "react";
import axios from "axios";

export default function App() {
  const videoRef = useRef();
  const [letter, setLetter] = useState("");
  const [word, setWord] = useState("");
  const backend = process.env.REACT_APP_BACKEND_URL;

  const captureAndSend = async () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = 640; canvas.height = 480;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg"));
    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    try {
      const resp = await axios.post(`${backend}/infer/`, form);
      setLetter(resp.data.letter);
      setWord(resp.data.word);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { videoRef.current.srcObject = stream; });
    const id = setInterval(captureAndSend, 500);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <video ref={videoRef} width="640" height="480" autoPlay />
      <div style={{ marginTop: 10 }}>Letter: <b>{letter}</b></div>
      <div>Word: <b>{word}</b></div>
    </div>
  );
}
