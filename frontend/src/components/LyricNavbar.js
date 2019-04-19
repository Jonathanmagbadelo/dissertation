import React from 'react';
import {Navbar, Nav} from 'react-bootstrap';

class LyricNavbar extends React.Component {
	render() {
		return (
			<Navbar bg="#005662" variant="dark" style={{backgroundColor: '#005662'}} expand='md'>
				<Navbar.Brand href="/">SONGIFAI</Navbar.Brand>
				<Nav className="mr-auto">
					<Nav.Link href="/lyric">New</Nav.Link>
				</Nav>
				<Nav className="ml-auto">
					<Nav.Link href="/about">About</Nav.Link>
				</Nav>
			</Navbar>
		);
	};
}

export {LyricNavbar};