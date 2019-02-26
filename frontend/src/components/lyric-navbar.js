import React from 'react';
import {Navbar, Nav} from 'react-bootstrap';
import {IoMdHome, IoIosAddCircleOutline} from 'react-icons/io';

class LyricNavbar extends React.Component{
	render() {
		return (
			<Navbar bg="dark" variant="dark" fixed="top">
				<Navbar.Brand href="/"><IoMdHome/></Navbar.Brand>
				<Nav className="ml-auto">
						<Nav.Link href="/"><IoIosAddCircleOutline/></Nav.Link>
				</Nav>
			</Navbar>
		);
	};
}

export {LyricNavbar};