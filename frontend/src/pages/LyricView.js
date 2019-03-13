import React from 'react';
import axios from "axios";

export default class ShowPage extends React.Component {

	state = {
		lyric: {}
	};

	componentDidMount() {
		this.getLyric(this.props.match.params.id)
	}

	getLyric = (lyric_id) => {
		axios
			.get("/api/lyrics/?id="+lyric_id)
			.then(result => this.setState({lyric: result.data[0]}));
	};

	render() {
		return (
			<div>
				<h1>{this.state.lyric.title}</h1>
				<p style={{whiteSpace:"pre"}}>{this.state.lyric.content}</p>
			</div>
		);
	}
}