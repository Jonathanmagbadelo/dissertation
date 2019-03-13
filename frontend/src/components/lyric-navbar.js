import React from 'react';
import {Navbar, Nav, Button} from 'react-bootstrap';
import Switch from "react-switch";

class LyricNavbar extends React.Component {
    state = {
        checked: true
    };

    handleChange = () => {
        this.setState({checked: !this.state.checked})
    };

    render() {
        return (
            <Navbar bg="#005662" variant="dark" style={{backgroundColor: '#005662'}}>
                <Nav>
                    <Nav.Item><Nav.Link href="/"><Button  variant="outline-info">Home</Button></Nav.Link></Nav.Item>
                </Nav>
                <Nav className="ml-auto">
                    <Nav.Item><Nav.Link>{this.switch()}</Nav.Link></Nav.Item>
                    <Nav.Item><Nav.Link>{this.switch()}</Nav.Link></Nav.Item>
                    <Nav.Item><Nav.Link href="/new"><Button  variant="outline-info">New</Button></Nav.Link></Nav.Item>
                </Nav>
            </Navbar>
        );
    };

    switch = () => {
        return (
            <label htmlFor="material-switch">
                <Switch
                    checked={this.state.checked}
                    onChange={this.handleChange}
                    onColor="#4fb3bf"
                    onHandleColor="#00838f"
                    handleDiameter={30}
                    uncheckedIcon={false}
                    checkedIcon={false}
                    boxShadow="0px 1px 5px rgba(0, 0, 0, 0.6)"
                    activeBoxShadow="0px 0px 1px 10px rgba(0, 0, 0, 0.2)"
                    height={20}
                    width={48}
                    className="react-switch"
                    id="material-switch"
                />
            </label>
        )
    }
}

export {LyricNavbar};