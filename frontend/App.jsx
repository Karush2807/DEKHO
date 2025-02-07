import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import CameraFeed from "../components/camera_feed";
import Alerts from "../components/alerts";
import MapView from "../components/mapview";

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
