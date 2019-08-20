import React from "react";
import Resizer from "react-image-file-resizer";
import axios from "axios";

const APIUrl = "https://baroque-saucisson-79648.herokuapp.com/prediction";

class ImageUploader extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      file: null,
      isLoading: false,
      result: null,
      isUploading: false
    };
    this.onFormSubmit = this.onFormSubmit.bind(this);
    this.onChange = this.onChange.bind(this);
  }
  onFormSubmit(e) {
    e.preventDefault();
    const formData = new FormData();
    formData.append("myImage", this.state.file);
    this.setState({ isLoading: true });
    const config = {
      headers: {
        "content-type": "image/jpeg"
      }
    };
    axios
      .post(APIUrl, this.state.file, config)
      .then(response => {
        this.setState({ result: response.data });
        this.setState({ isLoading: false });
      })
      .catch(error => {});
  }
  onChange(e) {
    this.setState({ isUploading: true });
    Resizer.imageFileResizer(
      e.target.files[0],
      1000,
      1000,
      "JPEG",
      100,
      0,
      uri => {
        const strImage = uri.replace(/^data:image\/[a-z]+;base64,/, "");
        this.setState({ file: strImage });
      },
      "base64"
    );
    this.setState({ isUploading: false });
  }

  render() {
    const spinner = (
      <div className="spinner-border" role="status">
        <span className="sr-only">Loading...</span>
      </div>
    );

    const result = this.state.result ? (
      <img src={"data:image/jpg;base64," + this.state.result} alt="failed to load" />
    ) : null;

    const showSpinnerOrResults = this.state.isLoading ? spinner : result;

    return (
      <div className="container-fluid my-5">
        <form onSubmit={this.onFormSubmit}>
          <input type="file" name="myImage" onChange={this.onChange} />
          <button type="submit" disabled={this.state.isUploading}>
            Upload
          </button>
        </form>
        <div className="container-fluid my-5">{showSpinnerOrResults}</div>
      </div>
    );
  }
}

export default ImageUploader;
