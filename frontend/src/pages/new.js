import React from 'react';
import {Form, Button, ButtonGroup} from 'react-bootstrap';
import axios from "axios";

export default class NewPage extends React.Component {
    state = {
        lyric: {
            title: '',
            content: '',
            createdAt: undefined,
            updatedAt: undefined,
            classification: undefined
        },
        suggested_words: []
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

    get_suggestions = (event) => {
        if( event.which === 32 ) {
            axios.get('api/assistant/suggest/').then(result => this.setState({suggested_words: result.data}));
        }
    };

    show_suggestions = () => {
        const words = Object.values(this.state.suggested_words).flat(1);
        return words.flatMap(word => <Button variant="secondary" size="lg" style={{border: '1px solid #4fb3bf'}}>{word}</Button>)
    };

    render() {
        return (
            <Form>
                <Form.Group controlId="formBasicEmail" align="center">
                    <Form.Label><h2>Title</h2></Form.Label>
                    <Form.Control id="title" type="email" placeholder="Enter title" onChange={this.updateValue}/>
                </Form.Group>

                <Form.Group controlId="exampleForm.ControlTextarea1" align="center">
                    <Form.Label><h2>Lyrics</h2></Form.Label>
                    <Form.Control id="content" as="textarea" rows="15" placeholder="Enter Lyrics..."
                                  onChange={this.updateValue} onKeyPress={this.get_suggestions}/>
                </Form.Group>

                <Form.Group align="center">
                    <ButtonGroup aria-label="Basic example">{this.show_suggestions()}</ButtonGroup>
                </Form.Group>

                <Form.Group align="center">
                    <Button variant="info" size="lg" onClick={this.post_lyric} style={{background: '#4fb3bf', color: 'black'}}>Post</Button>
                </Form.Group>

                <h4 align="right">Created At: {new Date().toLocaleDateString('en-GB')}</h4>
            </Form>
        );
    }
}