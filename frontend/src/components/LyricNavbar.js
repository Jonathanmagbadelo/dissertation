import React from 'react';
import {Navbar, Nav, DropdownButton, NavDropdown} from 'react-bootstrap';

class LyricNavbar extends React.Component {
	render() {
		return (
			<Navbar bg="#005662" variant="dark" style={{backgroundColor: '#005662'}} expand='md'>
				<Navbar.Brand href="/">SONGIFAI</Navbar.Brand>
				<Nav  className="mr-auto">
					<Nav.Item><Nav.Link href="/lyric">New</Nav.Link></Nav.Item>
					<Nav.Item>
						<NavDropdown title="Embedding" id="nav-dropdown">
							<NavDropdown.Item eventKey="4.1">Pop</NavDropdown.Item>
							<NavDropdown.Item eventKey="4.2">Rock</NavDropdown.Item>
							<NavDropdown.Item eventKey="4.3">Hip Hop</NavDropdown.Item>
							<NavDropdown.Divider />
							<NavDropdown.Item eventKey="4.4" active>Base Embedding</NavDropdown.Item>
						</NavDropdown>
					</Nav.Item>
				</Nav>
				<Nav className="ml-auto">
					<Nav.Item><Nav.Link href="/about">About</Nav.Link></Nav.Item>
				</Nav>
			</Navbar>
		);
	};
}

export {LyricNavbar};