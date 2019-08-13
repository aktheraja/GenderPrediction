import React from "react";

export default function NavBar() {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <span className="navbar-brand" href="#">
        Image Based Gender Predictor
      </span>
      <button
        className="navbar-toggler"
        type="button"
        data-toggle="collapse"
        data-target="#navbarSupportedContent"
        aria-controls="navbarSupportedContent"
        aria-expanded="false"
        aria-label="Toggle navigation"
      >
        <span className="navbar-toggler-icon" />
      </button>

      <div className="collapse navbar-collapse" id="navbarSupportedContent">
        <ul className="navbar-nav mr-auto">
          <li className="nav-item active">
            <a
              className="nav-link"
              href="https://github.com/savithcj/ImageBasedGenderPrediction"
              target="_blank"
              rel="noopener noreferrer"
            >
              Repo
            </a>
          </li>
          <li className="nav-item active">
            <a
              className="nav-link"
              href="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Dataset
            </a>
          </li>
        </ul>
      </div>
    </nav>
  );
}
