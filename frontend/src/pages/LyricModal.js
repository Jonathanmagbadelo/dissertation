import React from 'react';
import {Form, FormControl, Button, ButtonGroup, Card, ListGroup} from 'react-bootstrap';
import {Container, Col, Row, InputGroup} from 'react-bootstrap';
import axios from "axios";
import ToggleButtonGroup from 'react-bootstrap/ToggleButtonGroup'
import ToggleButton from 'react-bootstrap/ToggleButton'
import ButtonToolbar from 'react-bootstrap/ButtonToolbar'

export default class NewPage extends React.Component {

	new_lyric = {
		title: '',
		content: '',
		createdAt: undefined,
		updatedAt: undefined,
		classification: undefined
	};

	constructor(props) {
		super(props);
		this.state = {
			lyric: (props.location.state === undefined) ? this.new_lyric : props.location.state.lyric,
			predictions: [],
			suggestions: ["Suggestions will show here!"],
			edit: (props.location.state !== undefined),
			assist: false,
			rhyme: false,
			filter: false,
		};
		console.log("Test")
	}

	updateValue = (e) => {
		const {lyric} = this.state;

		this.setState({
			lyric: {...lyric, [e.target.id]: e.target.value}
		});
	};

	post_lyric = () => {
		axios('api/lyrics/new/', {
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

	update_lyric = () => {
		axios.put(`api/lyrics/${this.state.lyric.id}/`, this.state.lyric).then(response => {
			if (response.status === 200) {
				this.props.history.push("/");
			}
		})
			.catch(function (error) {
				console.log(error);
			});
	};

	save_lyric = () => {
		return this.state.edit ? this.update_lyric() : this.post_lyric()
	};

	get_predictions = (event) => {
		if (event.which === 32 && this.state.assist) {
			axios.get('api/assistant/predict/').then(result => this.setState({predictions: result.data}));
		}
	};

	get_suggestions = () => {
		const word = document.getElementById("suggest_word").value;
		axios.get(`api/assistant/suggest/?word=${word}`).then(result => this.setState({suggestions: result.data}));
	};

	showPredictions = () => {
		const words = Object.values(this.state.predictions).flat(1);
		return words.flatMap(word => <Button variant="secondary" size="lg"
											 style={{border: '1px solid #4fb3bf'}}>{word}</Button>)
	};

	handle_toggle = (event) => {
		const target = event.target.value;
		this.setState({[target]: !this.state[target]});
		if (!this.state.assist){
			this.setState({predictions: []});
			this.showPredictions();
		}
	};

	renderSuggestions(){
		const suggestions = Object.values(this.state.suggestions).flat(1);
		return suggestions.map(
			suggestion => <ListGroup.Item variant="secondary">{suggestion}</ListGroup.Item>
		);
	}

	render() {
		return (
			<Form>
				<br/>
				<Form.Group controlId="formBasicEmail" align="center">
					<Form.Label><h2>Title</h2></Form.Label>
					<Form.Control id="title" type="email" placeholder="Enter title..." onChange={this.updateValue}
								  style={{border: '3px solid #005662'}} value={this.state.lyric.title}/>
				</Form.Group>

				<Form.Group controlId="exampleForm.ControlTextarea1" align="center">
					<Form.Label><h2>Lyrics</h2></Form.Label>
					<Container>
						<Row>
							<Col>
								<ButtonToolbar>
									<ToggleButtonGroup type="checkbox" defaultValue={[]}>
										<ToggleButton value="assist" onChange={this.handle_toggle} style={{border: '3px solid #005662'}} >ASSIST</ToggleButton>
										<ToggleButton value="rhyme" onChange={this.handle_toggle} style={{border: '3px solid #005662'}} >RHYME</ToggleButton>
										<ToggleButton value="filter" onChange={this.handle_toggle} style={{border: '3px solid #005662'}} >FILTER</ToggleButton>
									</ToggleButtonGroup>
								</ButtonToolbar>
							</Col>
						</Row>
						<Row>
							<Col sm={8}>
								<Form.Control id="content" as="textarea" rows="15" placeholder="Enter Lyrics..."
											  onChange={this.updateValue} onKeyPress={this.get_predictions}
											  style={{border: '3px solid #005662'}} value={this.state.lyric.content}/>
							</Col>
							<Col>
								<Card style={{ width: '18rem' }}>
									<Card.Header>
										<InputGroup className="mb-3">
											<FormControl
												placeholder="Suggest Word..."
												id="suggest_word"
											/>
											<InputGroup.Append>
												<Button onClick={this.get_suggestions} variant="outline-success">Search</Button>
											</InputGroup.Append>
										</InputGroup>
										<ListGroup variant="flush">
											{this.renderSuggestions()}
										</ListGroup>
									</Card.Header>
								</Card>;
							</Col>
						</Row>
					</Container>
				</Form.Group>

				<Form.Group align="center">
					<ButtonGroup aria-label="Basic example">{this.showPredictions()}</ButtonGroup>
				</Form.Group>

				<Form.Group align="center">
					<Button variant="info" size="lg" onClick={this.save_lyric}
							style={{background: '#4fb3bf', color: 'black', border: '3px solid #005662'}}>Save</Button>
				</Form.Group>

				<h4 align="right">Updated At: {new Date().toLocaleDateString('en-GB')}</h4>
			</Form>
		);
	}
}