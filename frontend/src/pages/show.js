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
		let lyric = axios
			.get("/api/lyrics/?id="+lyric_id)
			.then(result => this.setState({lyric: result.data[0]}));

		return lyric;
	};

	render() {
		return (
			<div>
				<h1>{this.state.lyric.title}</h1>
				<div>{this.state.lyric.content}</div>
			</div>
		);
	}
}