import React, {Component} from 'react';
import './App.css';
import Main from "./Main";

export default class App extends Component {
  componentDidMount() {
    document.body.style.background = '#F7F7F7';
  }

  render = () => (
    <div className="App">
      <Main/>
    </div>
  );
}
