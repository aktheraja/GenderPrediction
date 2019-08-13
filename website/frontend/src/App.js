import React from "react";
import "./App.css";

import NavBar from "./components/NavBar";
import Info from "./components/Info";
import ImageUploader from "./components/ImageUploader";

function App() {
  return (
    <div className="App">
      <NavBar />
      <Info />
      <ImageUploader />
    </div>
  );
}

export default App;
