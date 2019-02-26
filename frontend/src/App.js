import React, {Component} from 'react';
import {BrowserRouter, Route} from 'react-router-dom';
import './App.css';

import IndexPage from './pages/index';
import ShowPage from './pages/show';
import {LyricNavbar} from './components/lyric-navbar'

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
				_id: 2,
				title: "Second Song",
				body: "Gucci Gang 2",
				updatedAt: new Date()
			}
		}
	};
	render() {
		return (
			<BrowserRouter>
				<div className="App">
					<LyricNavbar/>
					<Route exact path="/" component={(props) => <IndexPage {...props} lyrics={this.state.lyrics}/>}/>
					<Route exact path="/lyrics/:id" component={(props) => <ShowPage {...props} lyric={this.state.lyrics[props.match.params.id]}/>}/>
				</div>
			</BrowserRouter>
		);
	}
}

export default App;
