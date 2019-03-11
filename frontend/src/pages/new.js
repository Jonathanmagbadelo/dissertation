import React from 'react';
import {Form, Button, Modal, ButtonGroup} from 'react-bootstrap';
import axios from "axios";

export default class NewPage extends React.Component {
    state = {
        lyric: {
            title: '',
            content: '',
            createdAt: undefined,
            updatedAt: undefined,
            classification: undefined
        }
    };

    updateValue = (e) => {
        const {lyric} = this.state;

        this.setState({
            lyric: {...lyric, [e.target.id]: e.target.value}
        });
    };

    post_lyric = () => {
        axios('api/lyrics/new', {
            method: 'POST',
            data: this.state.lyric,
            withCredentials: true,
        })
            .then(response => {
                if (response.status === 201) {
                    this.props.history.push("/");
                }
            })
            .catch(function (error) {
                console.log(error);
            });
    };

    render() {
        return (
            <Form>
                <Form.Group controlId="formBasicEmail" align="center">
                    <Form.Label><h2>Title</h2></Form.Label>
                    <Form.Control type="email" placeholder="Enter title"/>
                </Form.Group>

                <Form.Group controlId="exampleForm.ControlTextarea1" align="center">
                    <Form.Label><h2>Lyrics</h2></Form.Label>
                    <Form.Control as="textarea" rows="15" placeholder="Enter Lyrics..."/>
                </Form.Group>

                <Form.Group align="center">
                    <ButtonGroup aria-label="Basic example">
                        <Button variant="secondary" size="lg">Left</Button>
                        <Button variant="secondary" size="lg">Middle</Button>
                        <Button variant="secondary" size="lg">Right</Button>
                    </ButtonGroup>
                </Form.Group>

                <Form.Group align="center">
                    <Button variant="info" size="lg">Post</Button>
                </Form.Group>

                <h4 align="right">Created At: {new Date().toLocaleDateString('en-GB')}</h4>
            </Form>
        );
    }
}