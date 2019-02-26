import React from 'react';

export default class ShowPage extends React.Component {
	render() {
		const {lyric} = this.props;

		return (
			<div>
				<h1>{lyric.title}</h1>
				<div>{lyric.body}</div>
			</div>
		);
	}
}