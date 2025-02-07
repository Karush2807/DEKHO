import React, { useState, useEffect } from "react";
import axios from "axios";

const CameraFeed = () => {
  const [frame, setFrame] = useState(null);

  useEffect(() => {
    const fetchFrame = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/video_feed", {
          responseType: "arraybuffer",
        });
        const imageUrl = URL.createObjectURL(new Blob([response.data]));
        setFrame(imageUrl);
      } catch (error) {
        console.error("Error fetching video feed", error);
      }
    };

    fetchFrame();
    const interval = setInterval(fetchFrame, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h2>Live Camera Feed</h2>
      {frame && <img src={frame} alt="Live Feed" />}
    </div>
  );
};

export default CameraFeed;
