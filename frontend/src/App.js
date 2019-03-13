import React, {Component} from 'react';
import {BrowserRouter, Route} from 'react-router-dom';
import './App.css';
import {Container} from 'react-bootstrap'

import AboutPage from './pages/About';
import IndexPage from './pages/Index';
import LyricView from './pages/LyricView';
import LyricModal from './pages/LyricModal';
import {LyricNavbar} from './components/LyricNavbar'

class App extends Component {
	render() {
		return (
			<BrowserRouter>
				<switch>
					<div>
						<LyricNavbar/>
						<Container>
							<Route exact path="/about" component={AboutPage}/>
							<Route exact path="/" component={IndexPage}/>
							<Route exact path="/lyrics/:id" component={LyricView}/>
							<Route exact path="/lyric" component={LyricModal}/>
						</Container>
					</div>
				</switch>
			</BrowserRouter>
		);
	}
}

export default App;
