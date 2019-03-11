import React from 'react';
import {Form, Button, Modal} from 'react-bootstrap';
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
                if (response.status === 201){
                    this.props.history.push("/");
                }
            })
            .catch(function (error) {
                console.log(error);
            });
    };

    render() {
        return (
            <Modal.Dialog>
                <Modal.Header>
                    <Modal.Title>New Lyric</Modal.Title>
                    <h6>Created At: {new Date().toLocaleDateString('en-GB')}</h6>
                </Modal.Header>

                <Modal.Body>
                    <Form>
                        <Form.Group>
                            <Form.Label>Title</Form.Label>
                            <Form.Control id="title" onChange={this.updateValue} type="email"
                                          placeholder="Enter Lyric Title..."/>
                        </Form.Group>
                        <Form.Group>
                            <Form.Label>Lyrics</Form.Label>
                            <Form.Control id="content" onChange={this.updateValue} as="textarea" rows="15"
                                          placeholder="Enter Lyrics..."/>
                        </Form.Group>
                    </Form>
                </Modal.Body>

                <Modal.Footer>
                    <Button variant="secondary" href="/">Close</Button>
                    <Button variant="primary" onClick={this.post_lyric}>Save changes</Button>
                </Modal.Footer>
            </Modal.Dialog>
        );
    }
}