import React from 'react';
import {Link} from 'react-router-dom';
import {ListGroup} from 'react-bootstrap';

export default class LyricsList extends React.Component {
	renderLyrics() {
		const lyrics = Object.values((this.props.lyrics));

		return lyrics.map(lyric => <ListGroup.Item variant="light" align="center"><Link to={`/lyrics/${lyric._id}`}>{lyric.title}</Link></ListGroup.Item>);
	}

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