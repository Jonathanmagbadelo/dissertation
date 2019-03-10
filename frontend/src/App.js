import React, {Component} from 'react';
import {BrowserRouter, Route} from 'react-router-dom';
import './App.css';
import {Container, Row, Col} from 'react-bootstrap'

import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faHome, faPlusSquare } from '@fortawesome/free-solid-svg-icons'

import IndexPage from './pages/index';
import ShowPage from './pages/show';
import NewPage from './pages/new';
import {LyricNavbar} from './components/lyric-navbar'

library.add(faHome, faPlusSquare);

class App extends Component {
	render() {
		return (
			<BrowserRouter>
				<switch>
					<div>
						<LyricNavbar/>
						<Container>
							<Route exact path="/" component={IndexPage}/>
							<Route exact path="/lyrics/:id" component={ShowPage}/>
							<Route exact path="/new" component={NewPage}/>
						</Container>
					</div>
				</switch>
			</BrowserRouter>
		);
	}
}

export default App;
