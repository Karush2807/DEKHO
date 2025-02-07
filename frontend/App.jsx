import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import CameraFeed from "./components/camera_feed"; // ✅ Fixed
import Alerts from "./components/alerts"; // ✅ Fixed
import MapView from "./components/mapview"; // ✅ Fixed

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<CameraFeed />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/map" element={<MapView />} />
      </Routes>
    </Router>
  );
}

export default App;
