import React, { Component } from 'react';
import {BrowserRouter, Route } from 'react-router-dom';
import logo from './logo.svg';
import './App.css';

import IndexPage from './pages/index';

class App extends Component {
  state = {
    lyrics: {
      1: {
        _id: 1,
        title: "First Song",
        body: "Gucci Gang",
        updatedAt: new Date()
      },
      2: {
        _id: 1,
        title: "Second Song",
        body: "Gucci Gang 2",
        updatedAt: new Date()
      }
    }
  }

  render() {
    return (
      <div className="App">
        <IndexPage lyrics={this.state.lyrics}/>
      </div>
    );
  }
}

export default App;
