import React from 'react';
import {Navbar, Nav} from 'react-bootstrap';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'

class LyricNavbar extends React.Component{
	render() {
		return (
			<Navbar bg="dark" variant="dark">
				<Navbar.Brand href="/api"><FontAwesomeIcon icon="home"/></Navbar.Brand>
				<Nav className="ml-auto">
						<Nav.Link href="/new"><FontAwesomeIcon icon="plus-square"/></Nav.Link>
				</Nav>
			</Navbar>
		);
	};
}

export {LyricNavbar};