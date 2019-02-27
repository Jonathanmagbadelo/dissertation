import React from 'react';
import {Form, Button, Modal} from 'react-bootstrap';

export default class NewPage extends React.Component {
	state = {
		lyric: {
			title: '',
			body: '',
			createdAt: undefined,
			updatedAt: undefined,
			classification: undefined
		}
	};

	handleSave = (e) => {
		e.preventDefault();

		const id = this.props.onSave(this.state.lyric);


		//alert("Lyric Title " + lyric.title + " Lyric Content " + lyric.body);
	};

	updateValue = (e) => {
		const {lyric} = this.state;

		this.setState({
			lyric: {...lyric, [e.target.id]: e.target.value}
		});
	};

	render() {
		const {lyric} = this.state;

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
							<Form.Control id="title" onChange={this.updateValue} type="email" placeholder="Enter Lyric Title..."/>
						</Form.Group>
						<Form.Group>
							<Form.Label>Lyrics</Form.Label>
							<Form.Control id="body" onChange={this.updateValue} as="textarea" rows="15" placeholder="Enter Lyrics..." />
						</Form.Group>
					</Form>
				</Modal.Body>

				<Modal.Footer>
					<Button variant="secondary" href="/">Close</Button>
					<Button variant="primary" onClick={this.handleSave}>Save changes</Button>
				</Modal.Footer>
			</Modal.Dialog>
		);
	}
}