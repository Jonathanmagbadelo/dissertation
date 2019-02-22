import React from 'react';
import {Link} from 'react-router-dom';

export default class LyricsList extends React.Component {
	renderLyrics() {
		const lyrics = Object.values((this.props.lyrics));

		return lyrics.map(lyric => <div><h2><Link to={"/lyric/${lyric._id}"}>{lyric.title}</Link></h2></div>);
	}

	render() {
		return (
			<div>
				{this.renderLyrics()}
			</div>
		)
	}

}