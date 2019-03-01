import React from 'react';
import {Link} from 'react-router-dom';
import {ListGroup} from 'react-bootstrap';
import axios from "axios";

export default class LyricsList extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			lyricsList: []
		};
		this.refreshList();
	}

	renderLyrics() {
		const lyrics = Object.values((this.state.lyricsList));
		return lyrics.flatMap(lyric => <ListGroup.Item variant="light" align="center"><Link to={`/lyrics/${lyric._id}`}>{lyric.title}</Link></ListGroup.Item>);
	}

	refreshList = () => {
		axios
			.get("/api/lyrics/")
			.then(res => this.setState({ LyricsList: res.data }))
			.catch(err => console.log(err));
	};

	render() {
		return (
			<div>
				<ListGroup>
					{this.renderLyrics()}
				</ListGroup>
			</div>
		)
	}

}