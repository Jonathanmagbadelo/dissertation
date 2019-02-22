import React, {Component} from 'react';
import {BrowserRouter, Route} from 'react-router-dom';
import logo from './logo.svg';
import './App.css';

import IndexPage from './pages/index';
import ShowPage from './pages/show';

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
			<BrowserRouter>
				<div className="App">
					<Route exact path="/" component={(props) => <IndexPage {...props} lyrics={this.state.lyrics}/>}/>
					<Route exact path="/lyric/:id"
						   component={(props) => <ShowPage {...props} lyric={this.state.lyrics[props.match.params.id]}/>}/>
				</div>
			</BrowserRouter>
		);
	}
}

export default App;
